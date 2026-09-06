"""Gauss-Hermite nodes and weights, transformed for a standard normal measure."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class Quadrature(NamedTuple):
	nodes: Array
	log_weights: Array


def gauss_hermite(n_nodes: int) -> Quadrature:
	"""A rule whose weighted sum approximates E[f(theta)] under theta ~ Normal (0, 1)."""
	if n_nodes < 1:
		raise ValueError(f"n_nodes must be positive, got {n_nodes}")

	x, w = np.polynomial.hermite.hermgauss(n_nodes)
	return Quadrature(
		nodes=jnp.asarray(np.sqrt(2.0) * x),
		log_weights=jnp.asarray(np.log(w) - 0.5 * np.log(np.pi)),
	)
