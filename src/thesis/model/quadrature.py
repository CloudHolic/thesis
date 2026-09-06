"""STandard Gauss-Hermite nodes and weights, left untransformed."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class Quadrature(NamedTuple):
	"""The raw Hermite rule for the weight exp(-x^2)."""

	x: Array
	log_w: Array


def gauss_hermite(n_nodes: int) -> Quadrature:
	"""A rule whose weighted sum approximates E[f(theta)] under theta ~ Normal (0, 1)."""
	if n_nodes < 1:
		raise ValueError(f"n_nodes must be positive, got {n_nodes}")

	x, w = np.polynomial.hermite.hermgauss(n_nodes)
	return Quadrature(x=jnp.asarray(x), log_w=jnp.asarray(np.log(w)))
