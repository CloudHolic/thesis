"""Per-person posterior mode and curvature width, which position the adaptive nodes."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

NEWTON_STEPS = 6
MAX_STEP = 2.0


def fit(log_h: Callable[[Array], Array], n_persons: int) -> tuple[Array, Array]:
	"""Newton on each person's log integrand, vectorized over persons."""
	first = jax.grad(lambda t: log_h(t).sum())
	second = jax.grad(lambda t: first(t).sum())

	theta = jnp.zeros(n_persons)
	for _ in range(NEWTON_STEPS):
		g1 = first(theta)
		g2 = second(theta)
		step = jnp.where(g2 < 0.0, g1 / g2, 0.0)
		theta = theta - jnp.clip(step, -MAX_STEP, MAX_STEP)

	curvature = second(theta)
	usable = curvature < 0.0

	return (
		jnp.where(usable, theta, 0.0),
		jnp.where(usable, jax.lax.rsqrt(jnp.where(usable, -curvature, 1.0)), 1.0),
	)
