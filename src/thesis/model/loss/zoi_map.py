"""The MAP objective: marginal likelihood plus the z-space prior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model import likelihood
from thesis.model.family.registry import FAMILIES
from thesis.utils.quadrature import Quadrature

_FAMILY = FAMILIES["zoi_beta"]
_LOG_SQRT_2PI = 0.9189385332046727


def init(response: np.ndarray, n_items: int) -> np.ndarray:
	"""Starting coordinates. A point estimate needs nothing but z."""
	return _FAMILY.initial_z(response, n_items)


def log_prior(z: Array) -> Array:
	"""Independent Normal (0, prior_scale) over z."""
	scale = _FAMILY.prior_scale

	return -(0.5 * jnp.square(z / scale) + jnp.log(scale) + _LOG_SQRT_2PI).sum()


def loss(
	z: Array,
	quad: Quadrature,
	responses: likelihood.Responses,
	key: Array | None = None,
) -> Array:
	"""Negative log posterior over the item coordinates. Deterministic, so `key` is unused."""
	del key
	tau = _FAMILY.to_tau(z)

	return -likelihood.log_marginal(_FAMILY, tau, quad, responses).sum() - log_prior(z)


def summary(z: np.ndarray, key: Array | None = None) -> dict[str, np.ndarray]:
	"""What the fit reports. A point estimate has no interval to report."""
	del key
	return {"z": np.asarray(z)}
