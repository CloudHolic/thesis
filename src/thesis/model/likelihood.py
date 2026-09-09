"""The ZOI Beta-IRT marginal likelihood: z to tau, the kernel, and adaptive quadrature.

One computation in stages -- transform, branch kernel, Hermite rule, Laplace centring,
person-major marginalization -- so they live together rather than in six files.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import stop_gradient
from jax.nn import log_sigmoid
from jax.ops import segment_sum
from jax.scipy.special import gammaln, logsumexp

Z_DIM = 5

NEWTON_STEPS = 6
MAX_STEP = 2.0
TOLERANCE = 1e-3

_LOG2 = 0.6931471805599453
_SQRT2 = 1.4142135623730951
_LOG_SQRT_2PI = 0.9189385332046727


# --- the unconstrained coordinate and its image ---------------------------------


class Tau(NamedTuple):
	"""Item parameters as the likelihood uses them."""

	a: Array
	b: Array
	omega: Array
	gamma_0: Array
	gamma_1: Array


def to_tau(z: Array) -> Tau:
	"""Maps z = (log a, b, omega, gamma_0, lambda) to tau."""
	if z.shape[-1] != Z_DIM:
		raise ValueError(f"z must have last dimension {Z_DIM}, got {z.shape[-1]}")

	return Tau(
		a=jnp.exp(z[..., 0]),
		b=z[..., 1],
		omega=z[..., 2],
		gamma_0=z[..., 3],
		gamma_1=z[..., 3] + jnp.exp(z[..., 4]),
	)


# --- log-space primitives JAX does not provide ----------------------------------


def log1mexp(u: Array) -> Array:
	"""log(1 - exp(u)) for u < 0."""
	u = jnp.minimum(u, -jnp.finfo(jnp.asarray(u).dtype).tiny)
	near = u > -_LOG2

	return jnp.where(
		near,
		jnp.log(-jnp.expm1(jnp.where(near, u, -1.0))),
		jnp.log1p(-jnp.exp(jnp.where(near, -1.0, u))),
	)


def log_diff_exp(high: Array, low: Array) -> Array:
	"""log(exp(high) - exp(low)), requiring high > low."""
	return high + log1mexp(low - high)


# --- the Hermite rule, left untransformed ---------------------------------------


class Quadrature(NamedTuple):
	"""The raw Hermite rule for the weight exp(-x^2)."""

	x: Array
	log_w: Array


def gauss_hermite(n_nodes: int) -> Quadrature:
	"""The rule as tabulated. Placing it on a normal measure is the caller's job."""
	if n_nodes < 1:
		raise ValueError(f"n_nodes must be positive, got {n_nodes}")

	x, w = np.polynomial.hermite.hermgauss(n_nodes)
	return Quadrature(x=jnp.asarray(x), log_w=jnp.asarray(np.log(w)))


# --- the kernel, per response at that person's nodes ----------------------------


def gather(tau: Tau, items: Array) -> Tau:
	"""Turns per-item fields into per-response ones by picking each response's item."""
	return Tau(*(field[items] for field in tau))


def log_k_interior(tau: Tau, theta: Array, log_y: Array, log1m_y: Array) -> Array:
	"""log k inside (0, 1). `tau` is gathered; theta and the result are (n, n_nodes)."""
	a_theta = tau.a[:, None] * theta
	eta = a_theta + tau.b[:, None]
	half_omega = tau.omega[:, None] / 2.0

	alpha = jnp.exp(eta / 2.0 + half_omega)
	beta = jnp.exp(-eta / 2.0 + half_omega)
	log_pi_b = log_diff_exp(
		log_sigmoid(tau.gamma_1[:, None] - a_theta),
		log_sigmoid(tau.gamma_0[:, None] - a_theta),
	)

	return (
		log_pi_b
		+ (alpha - 1.0) * log_y[:, None]
		+ (beta - 1.0) * log1m_y[:, None]
		- (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
	)


def log_k_zero(tau: Tau, theta: Array) -> Array:
	"""log k at exactly 0. The Beta shapes are never formed."""
	return log_sigmoid(tau.gamma_0[:, None] - tau.a[:, None] * theta)


def log_k_one(tau: Tau, theta: Array) -> Array:
	"""log k at exactly 1. The Beta shapes are never formed."""
	return log_sigmoid(tau.a[:, None] * theta - tau.gamma_1[:, None])


# --- where each person's nodes go -----------------------------------------------


class Center(NamedTuple):
	"""Where each person's nodes go, and whether Newton earned the right to place them."""

	mode: Array
	sd: Array
	usable: Array


def newton_center(log_h: Callable[[Array], Array], n_persons: int) -> Center:
	"""Newton on each person's log integrand, vectorized over persons."""
	first = jax.grad(lambda t: log_h(t).sum())
	second = jax.grad(lambda t: first(t).sum())

	theta = jnp.zeros(n_persons)
	for _ in range(NEWTON_STEPS):
		g1 = first(theta)
		g2 = second(theta)
		concave = g2 < 0.0
		step = jnp.where(concave, g1 / jnp.where(concave, g2, -1.0), -jnp.sign(g1) * MAX_STEP)
		theta = theta - jnp.clip(step, -MAX_STEP, MAX_STEP)

	curvature = second(theta)
	concave = curvature < 0.0
	settled = jnp.abs(first(theta)) < TOLERANCE * jnp.sqrt(jnp.where(concave, -curvature, 1.0))
	usable = concave & settled

	return Center(
		mode=jnp.where(usable, theta, 0.0),
		sd=jnp.where(usable, jax.lax.rsqrt(jnp.where(usable, -curvature, 1.0)), 1.0),
		usable=usable,
	)


# --- person-major marginalization -----------------------------------------------


class Responses(NamedTuple):
	"""Responses partitioned by mixture branch, so the Beta term never sees y=0 or y=1."""

	n_persons: int  # a branch can be empty and then cannot report the count
	interior_item: Array
	interior_person: Array
	interior_log_y: Array
	interior_log1m_y: Array
	zero_item: Array
	zero_person: Array
	one_item: Array
	one_person: Array


def split_by_branch(
	item_index: np.ndarray, person_index: np.ndarray, response: np.ndarray, n_persons: int
) -> Responses:
	"""Partitions responses into the interior, zero and one branches, preserving order."""
	if not ((response >= 0.0) & (response <= 1.0)).all():
		raise ValueError("response must be in [0, 1]")

	at_zero = response == 0.0
	at_one = response == 1.0
	interior = ~(at_zero | at_one)
	y = response[interior]

	return Responses(
		n_persons=n_persons,
		interior_item=jnp.asarray(item_index[interior]),
		interior_person=jnp.asarray(person_index[interior]),
		interior_log_y=jnp.asarray(np.log(y)),
		interior_log1m_y=jnp.asarray(np.log1p(-y)),
		zero_item=jnp.asarray(item_index[at_zero]),
		zero_person=jnp.asarray(person_index[at_zero]),
		one_item=jnp.asarray(item_index[at_one]),
		one_person=jnp.asarray(person_index[at_one]),
	)


def _log_likelihood(tau: Tau, responses: Responses, theta: Array) -> Array:
	"""Sum of log k over each person's responses. `theta` is (n_persons, n_nodes)."""
	n_persons = responses.n_persons

	totals = segment_sum(
		log_k_interior(
			gather(tau, responses.interior_item),
			theta[responses.interior_person],
			responses.interior_log_y,
			responses.interior_log1m_y,
		),
		responses.interior_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		log_k_zero(gather(tau, responses.zero_item), theta[responses.zero_person]),
		responses.zero_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		log_k_one(gather(tau, responses.one_item), theta[responses.one_person]),
		responses.one_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)

	return totals


def find_center(tau: Tau, responses: Responses) -> Center:
	"""Where each person's adaptive nodes belong, and who fell back to (0, 1)."""
	return newton_center(
		lambda t: _log_likelihood(tau, responses, t[:, None])[:, 0] - 0.5 * jnp.square(t),
		responses.n_persons,
	)


def log_marginal(
	tau: Tau, quad: Quadrature, responses: Responses, *, center: Center | None = None
) -> Array:
	"""Marginal log-likelihood per person, by Gauss-Hermite centred on each posterior."""
	found = find_center(tau, responses) if center is None else center
	mode = stop_gradient(found.mode)
	sd = stop_gradient(found.sd)

	theta = mode[:, None] + _SQRT2 * sd[:, None] * quad.x[None, :]

	# x^2 + log phi(theta) with the two canceling x^2 terms expanded apart,
	# so that at sd = 1 they annihilate exactly rather than losing digits to subtraction.
	weight = (
		jnp.square(quad.x)[None, :] * (1.0 - jnp.square(sd)[:, None])
		- _SQRT2 * (sd * mode)[:, None] * quad.x[None, :]
		- 0.5 * jnp.square(mode)[:, None]
		- _LOG_SQRT_2PI
	)

	return jnp.log(_SQRT2 * sd) + logsumexp(
		quad.log_w[None, :] + weight + _log_likelihood(tau, responses, theta), axis=1
	)
