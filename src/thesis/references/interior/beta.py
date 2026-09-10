"""The Beta interior, in Molenaar's (eta, omega) parameterization."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import gammaln

N_PARAMS: int = 3
TAU_NAMES: tuple[str, ...] = ("a", "b", "omega")
SITES: tuple[str, ...] = ("log_a", "b", "omega")


class BetaTau(NamedTuple):
	"""Interior parameters as the density uses them."""

	a: Array
	b: Array
	omega: Array


def interior_stats(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""What the density needs from the response and the caller can precompute once."""
	return np.log(y), np.log1p(-y)


def log_f(tau: BetaTau, theta: Array, stats: tuple[Array, ...]) -> Array:
	"""log of the Beta density, with the shapes formed from eta = a * theta + b."""
	log_y, log1m_y = stats
	eta = tau.a * theta + tau.b
	half_omega = tau.omega / 2.0

	alpha = jnp.exp(eta / 2.0 + half_omega)
	beta = jnp.exp(-eta / 2.0 + half_omega)

	return (
		(alpha - 1.0) * log_y
		+ (beta - 1.0) * log1m_y
		- (gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta))
	)
