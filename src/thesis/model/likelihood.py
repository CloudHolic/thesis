"""The ZOI Beta-IRT marginal likelihood: z to tau, the kernel, and adaptive quadrature."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.ops import segment_sum

from thesis.references.interior import beta
from thesis.references.zoi import kernel
from thesis.utils.quadrature import Center, Quadrature, expect_standard_normal, newton_center

Z_DIM = 5


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


# --- the kernel, per response at that person's nodes ----------------------------


def gather(tau: Tau, items: Array) -> Tau:
	"""Turns per-item fields into per-response ones by picking each response's item."""
	return Tau(*(field[items] for field in tau))


def log_k_interior(tau: Tau, theta: Array, log_y: Array, log1m_y: Array) -> Array:
	"""log k inside (0, 1). `tau` is gathered; theta and the result are (n, n_nodes)."""
	a = tau.a[:, None]

	return kernel.log_pi_b(a * theta, tau.gamma_0[:, None], tau.gamma_1[:, None]) + beta.log_f(
		beta.BetaTau(a=a, b=tau.b[:, None], omega=tau.omega[:, None]),
		theta,
		(log_y[:, None], log1m_y[:, None]),
	)


def log_k_zero(tau: Tau, theta: Array) -> Array:
	"""log k at exactly 0. The Beta shapes are never formed."""
	return kernel.log_k_zero(tau.a[:, None] * theta, tau.gamma_0[:, None])


def log_k_one(tau: Tau, theta: Array) -> Array:
	"""log k at exactly 1. The Beta shapes are never formed."""
	return kernel.log_k_one(tau.a[:, None] * theta, tau.gamma_1[:, None])


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

	return expect_standard_normal(lambda theta: _log_likelihood(tau, responses, theta), quad, found)
