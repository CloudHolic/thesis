"""The MAP objective: marginal likelihood plus the z-space prior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model import encoder, likelihood

PRIOR_SCALE = 10.0
_LOG_SQRT_2PI = 0.9189385332046727


def init(response: np.ndarray, n_items: int) -> np.ndarray:
	"""Starting coordinates. A point estimate needs nothing but z."""
	return encoder.initial_z(response, n_items)


def log_prior(z: Array) -> Array:
	"""Independent Normal (0, PRIOR_SCALE) over z."""
	return -(0.5 * jnp.square(z / PRIOR_SCALE) + jnp.log(PRIOR_SCALE) + _LOG_SQRT_2PI).sum()


def log_marginal_total(
	tau: likelihood.Tau, quad: likelihood.Quadrature, responses: likelihood.Responses
) -> Array:
	"""Marginal log-likelihood summed over persons."""
	return likelihood.log_marginal(tau, quad, responses).sum()


def loss(
	z: Array,
	quad: likelihood.Quadrature,
	responses: likelihood.Responses,
	key: Array | None = None,
) -> Array:
	"""Negative log posterior over the item coordinates. Deterministic, so `key` is unused."""
	del key
	tau = likelihood.to_tau(z)
	return -likelihood.log_marginal(tau, quad, responses).sum() - log_prior(z)


def summary(z: np.ndarray, key: Array | None = None) -> dict[str, np.ndarray]:
	"""What the fit reports. A point estimate has no interval to report."""
	del key
	return {"z": np.asarray(z)}
