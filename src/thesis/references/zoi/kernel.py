"""Molenaar's zero-and-one inflation, as a wrapper over any interior density."""

from __future__ import annotations

from jax import Array
from jax.nn import log_sigmoid

from thesis.utils.logspace import log_diff_exp


def log_k_zero(a_theta: Array, gamma_0: Array) -> Array:
	"""log k at exactly 0."""
	return log_sigmoid(gamma_0 - a_theta)


def log_k_one(a_theta: Array, gamma_1: Array) -> Array:
	"""log k at exactly 1."""
	return log_sigmoid(a_theta - gamma_1)


def log_pi_b(a_theta: Array, gamma_0: Array, gamma_1: Array) -> Array:
	"""log of the mass the interior branch carries."""
	return log_diff_exp(log_sigmoid(gamma_1 - a_theta), log_sigmoid(gamma_0 - a_theta))
