"""The MAP objective: marginal likelihood plus the z-space prior."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model import likelihood
from thesis.model.family.protocol import Family
from thesis.model.fit import Objective
from thesis.utils.quadrature import Quadrature

_LOG_SQRT_2PI = 0.9189385332046727


def build(family: Family) -> Objective:
	"""A point-estimate objective over `family`'s item coordinates."""
	scale = family.prior_scale

	def init(response: np.ndarray, n_items: int) -> np.ndarray:
		"""Starting coordinates. A point estimate needs nothing but z."""
		return family.initial_z(response, n_items)

	def log_prior(z: Array) -> Array:
		return -(0.5 * jnp.square(z / scale) + jnp.log(scale) + _LOG_SQRT_2PI).sum()

	def loss(
		z: Any,
		quad: Quadrature,
		responses: likelihood.Responses,
		key: Array | None = None,
	) -> Array:
		"""Negative log posterior over the item coordinates. Deterministic, so `key` is unused."""
		del key
		return -likelihood.log_marginal(
			family, family.to_tau(z), quad, responses
		).sum() - log_prior(z)

	def summary(z: Any, key: Array | None = None) -> dict[str, np.ndarray]:
		"""What the fit reports. A point estimate has no interval to report."""
		del key
		return {"z": np.asarray(z)}

	return Objective(init=init, loss=loss, summary=summary)
