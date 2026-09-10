"""Adaptive Gauss-Hermite: the Hermite rule, Newton mode-finding, and the expectation."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp

NEWTON_STEPS = 6
MAX_STEP = 2.0
TOLERANCE = 1e-3

_SQRT2 = 1.4142135623730951
_LOG_SQRT_2PI = 0.9189385332046727


class Quadrature(NamedTuple):
	"""The raw Hermite rule for the weight exp(-x^2)."""

	x: Array
	log_w: Array


class Center(NamedTuple):
	"""Where each row's nodes go, and whether Newton earned the right to place them."""

	mode: Array
	sd: Array
	usable: Array


def gauss_hermite(n_nodes: int) -> Quadrature:
	"""The rule as tabulated. Placing it on a normal measure is the caller's job."""
	if n_nodes < 1:
		raise ValueError(f"n_nodes must be positive, got {n_nodes}")

	x, w = np.polynomial.hermite.hermgauss(n_nodes)
	return Quadrature(x=jnp.asarray(x), log_w=jnp.asarray(np.log(w)))


def newton_center(log_h: Callable[[Array], Array], n_rows: int) -> Center:
	"""Newton on each row's log integrand, vectorized over rows."""
	first = jax.grad(lambda t: log_h(t).sum())
	second = jax.grad(lambda t: first(t).sum())

	theta = jnp.zeros(n_rows)
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


def expect_standard_normal(
	log_g: Callable[[Array], Array], quad: Quadrature, center: Center
) -> Array:
	"""log E[exp(log_g(t))] under t ~ Normal(0, 1), one value per row.

	`log_g` receives the nodes as (n_rows, n_nodes) and returns that same shape.
	"""
	mode = stop_gradient(center.mode)
	sd = stop_gradient(center.sd)

	theta = mode[:, None] + _SQRT2 * sd[:, None] * quad.x[None, :]

	# x^2 + log phi(theta) with the two canceling x^2 terms expanded apart,
	# so that at sd = 1 they annihilate exactly rather than losing digits to subtraction.
	weight = (
		jnp.square(quad.x)[None, :] * (1.0 - jnp.square(sd)[:, None])
		- _SQRT2 * (sd * mode)[:, None] * quad.x[None, :]
		- 0.5 * jnp.square(mode)[:, None]
		- _LOG_SQRT_2PI
	)

	return jnp.log(_SQRT2 * sd) + logsumexp(quad.log_w[None, :] + weight + log_g(theta), axis=1)
