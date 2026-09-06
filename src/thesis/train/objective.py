"""The MAP objective: marginal likelihood plus the z-space prior."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from thesis.model import marginal, transform
from thesis.model.quadrature import Quadrature

PRIOR_SCALE = 10.0
_LOG_SQRT_2PI = 0.9189385332046727


def log_prior(z: Array) -> Array:
	"""Independent Normal (0, PRIOR_SCALE) over z."""
	return -(0.5 * jnp.square(z / PRIOR_SCALE) + jnp.log(PRIOR_SCALE) + _LOG_SQRT_2PI).sum()


def log_marginal_total(
	tau: transform.Tau, quad: Quadrature, responses: marginal.Responses
) -> Array:
	"""Marginal log-likelihood summed over persons."""
	return marginal.log_marginal(tau, quad, responses).sum()


def map_loss(z: Array, quad: Quadrature, responses: marginal.Responses) -> Array:
	"""Negative log posterior over the item coordinates."""
	tau = transform.to_tau(z)
	return -log_marginal_total(tau, quad, responses) - log_prior(z)
