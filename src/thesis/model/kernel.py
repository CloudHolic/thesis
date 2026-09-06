"""The ZOI Beta-IRT kernel, evaluated per response at that person's nodes."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.nn import log_sigmoid
from jax.scipy.special import gammaln

from . import logutil
from .transform import Tau


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
	log_pi_b = logutil.log_diff_exp(
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
	"""log k at exactly 0."""
	return log_sigmoid(tau.gamma_0[:, None] - tau.a[:, None] * theta)


def log_k_one(tau: Tau, theta: Array) -> Array:
	"""log k at exactly 1."""
	return log_sigmoid(tau.a[:, None] * theta - tau.gamma_1[:, None])
