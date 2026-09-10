"""The Beta interior in the coordinates we fit it in."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.references.interior.beta import (
	N_PARAMS,
	SITES,
	TAU_NAMES,
	BetaTau,
	interior_stats,
	log_f,
)

__all__ = [
	"N_PARAMS",
	"SITES",
	"TAU_NAMES",
	"BetaTau",
	"b_star",
	"initial_z",
	"interior_stats",
	"log_f",
	"prior_scale",
	"quantities",
	"tau_from_sites",
	"to_tau",
]

prior_scale: float = 10.0


def to_tau(z: Array) -> BetaTau:
	"""Maps z = (log a, b, omega) to the interior parameters."""
	return BetaTau(a=jnp.exp(z[..., 0]), b=z[..., 1], omega=z[..., 2])


def tau_from_sites(means: dict[str, np.ndarray]) -> BetaTau:
	"""The same parameters from a reference posterior, which samples them directly."""
	return BetaTau(
		a=jnp.exp(jnp.asarray(means["log_a"])),
		b=jnp.asarray(means["b"]),
		omega=jnp.asarray(means["omega"]),
	)


def initial_z(response: np.ndarray) -> np.ndarray:
	"""Moment-matched start from the interior responses, shape (N_PARAMS,)."""
	interior = response[(response > 0.0) & (response < 1.0)]
	if interior.size == 0:
		raise ValueError("cannot initialize from responses with no interior mass")

	mean = float(interior.mean())
	var = float(interior.var())

	eta = float(np.log(mean / (1.0 - mean)))
	phi = mean * (1.0 - mean) / var - 1.0
	if phi <= 0.0:
		raise ValueError(f"interior responses are overdispersed for a Beta: phi={phi}")
	omega = 2.0 * float(np.log(phi / (2.0 * np.cosh(eta / 2.0))))

	return np.array([0.0, eta, omega])


def b_star(tau: BetaTau, y: float) -> Array:
	"""The ability whose conditional interior mean is y. Larger is harder."""
	return (np.log(y / (1.0 - y)) - tau.b) / tau.a


def quantities(tau: BetaTau) -> dict[str, np.ndarray]:
	"""The dispersion, which is read on its own scale."""
	return {"omega": np.asarray(tau.omega)}
