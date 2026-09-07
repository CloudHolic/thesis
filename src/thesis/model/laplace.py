"""Per-person posterior mode and curvature width, which position the adaptive nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

NEWTON_STEPS = 6
MAX_STEP = 2.0
TOLERANCE = 1e-3


class Center(NamedTuple):
	"""Where each person's nodes go, and whether Newton earned the right to place them."""

	mode: Array
	sd: Array
	usable: Array


def fit(log_h: Callable[[Array], Array], n_persons: int) -> Center:
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
