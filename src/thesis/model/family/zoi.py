"""Wrapping any interior density with Molenaar's two boundary atoms."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model.family.protocol import Family, Interior
from thesis.references.zoi import kernel

BOUNDARY_NAMES: tuple[str, ...] = ("gamma_0", "gamma_1")


class ZoiTau(NamedTuple):
	"""The interior's own parameters, plus the two boundary coordinates."""

	interior: Any
	gamma_0: Array
	gamma_1: Array


def wrap(interior: Interior) -> Family:
	"""A Family that adds the zero and one atoms to `interior`."""
	n_interior = interior.N_PARAMS
	z_dim = n_interior + 2

	def to_tau(z: Array) -> ZoiTau:
		if z.shape[-1] != z_dim:
			raise ValueError(f"z must have last dimension {z_dim}, got {z.shape[-1]}")
		gamma_0 = z[..., n_interior]

		return ZoiTau(
			interior=interior.to_tau(z[..., :n_interior]),
			gamma_0=gamma_0,
			gamma_1=gamma_0 + jnp.exp(z[..., n_interior + 1]),
		)

	def tau_from_sites(means: dict[str, np.ndarray]) -> ZoiTau:
		return ZoiTau(
			interior=interior.tau_from_sites(means),
			gamma_0=jnp.asarray(means["gamma_0"]),
			gamma_1=jnp.asarray(means["gamma_1"]),
		)

	def log_k_interior(tau: ZoiTau, theta: Array, stats: tuple[Array, ...]) -> Array:
		a_theta = tau.interior.a * theta

		return kernel.log_pi_b(a_theta, tau.gamma_0, tau.gamma_1) + interior.log_f(
			tau.interior, theta, stats
		)

	def log_k_zero(tau: ZoiTau, theta: Array) -> Array:
		return kernel.log_k_zero(tau.interior.a * theta, tau.gamma_0)

	def log_k_one(tau: ZoiTau, theta: Array) -> Array:
		return kernel.log_k_one(tau.interior.a * theta, tau.gamma_1)

	def initial_z(response: np.ndarray, n_items: int) -> np.ndarray:
		n = response.size
		if n == 0:
			raise ValueError("cannot initialize from an empty response array")

		start = interior.initial_z(response)

		# An empty boundary gets half an observation.
		floor = 0.5 / n
		p_zero = max(float((response == 0.0).mean()), floor)
		p_one = max(float((response == 1.0).mean()), floor)
		gamma_0 = float(np.log(p_zero / (1.0 - p_zero)))
		gamma_1 = float(np.log((1.0 - p_one) / p_one))
		if gamma_1 <= gamma_0:
			raise ValueError(f"boundary masses give gamma_1 <= gamma_0: {gamma_1} <= {gamma_0}")

		return np.tile(np.concatenate([start, [gamma_0, np.log(gamma_1 - gamma_0)]]), (n_items, 1))

	def b_star(tau: ZoiTau, y: float) -> Array:
		return interior.b_star(tau.interior, y)

	def quantities(tau: ZoiTau) -> dict[str, np.ndarray]:
		a = np.asarray(tau.interior.a)

		return {
			"gamma_1/a": np.asarray(tau.gamma_1) / a,
			"gamma_0/a": np.asarray(tau.gamma_0) / a,
			**interior.quantities(tau.interior),
		}

	return Family(
		Z_DIM=z_dim,
		TAU_NAMES=(*interior.TAU_NAMES, *BOUNDARY_NAMES),
		SITES=(*interior.SITES, *BOUNDARY_NAMES),
		prior_scale=interior.prior_scale,
		to_tau=to_tau,
		tau_from_sites=tau_from_sites,
		interior_stats=interior.interior_stats,
		log_k_zero=log_k_zero,
		log_k_interior=log_k_interior,
		log_k_one=log_k_one,
		initial_z=initial_z,
		b_star=b_star,
		quantities=quantities,
	)
