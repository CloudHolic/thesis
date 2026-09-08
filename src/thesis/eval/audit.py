"""Quadrature accuracy against direct numerical integration, kept as an artifact."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from thesis.model import kernel, laplace, marginal, quadrature
from thesis.model.transform import Tau

GRID_LIMIT = 8.0
GRID_POINTS = 4001
QUAD_LIMIT = 800


def log_integrand(tau: Tau, items: np.ndarray, ys: np.ndarray, thetas: np.ndarray) -> np.ndarray:
	"""log[prod_i k(y_i | theta) phi(theta)] for one person, at each theta."""

	def block(mask: np.ndarray) -> jnp.ndarray:
		return jnp.asarray(np.tile(thetas, (int(mask.sum()), 1)))

	total = np.asarray(norm.logpdf(thetas), dtype=float)
	interior = (ys > 0.0) & (ys < 1.0)
	if interior.any():
		y = ys[interior]
		total += np.asarray(
			kernel.log_k_interior(
				kernel.gather(tau, jnp.asarray(items[interior])),
				block(interior),
				jnp.asarray(np.log(y)),
				jnp.asarray(np.log1p(-y)),
			)
		).sum(axis=0)

	for mask, fn in ((ys == 0.0, kernel.log_k_zero), (ys == 1.0, kernel.log_k_one)):
		if mask.any():
			total += np.asarray(fn(kernel.gather(tau, jnp.asarray(items[mask])), block(mask))).sum(
				axis=0
			)

	return total


def exact(tau: Tau, items: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
	"""(log integral, posterior mode, posterior sd) by direct adaptive quadrature."""
	grid = np.linspace(-GRID_LIMIT, GRID_LIMIT, GRID_POINTS)
	vals = log_integrand(tau, items, ys, grid)
	peak = float(vals.max())
	mode = float(grid[int(vals.argmax())])

	weights = np.exp(vals - peak)
	weights /= weights.sum()
	centre = float((weights * grid).sum())
	sd = float(np.sqrt((weights * (grid - centre) ** 2).sum()))

	value, _err = quad(
		lambda t: math.exp(float(log_integrand(tau, items, ys, np.array([t]))[0]) - peak),
		-12.0,
		12.0,
		limit=QUAD_LIMIT,
		points=[mode],
	)

	return math.log(value) + peak, mode, sd


def estimate(tau: Tau, items: np.ndarray, ys: np.ndarray, n_nodes: int, *, adaptive: bool) -> float:
	"""Out log marginal for one person, with the nodes solved for or pinned at (0, 1)."""
	split = marginal.split_by_branch(items, np.zeros(items.size, dtype=int), ys, 1)
	center = (
		None
		if adaptive
		else laplace.Center(mode=jnp.zeros(1), sd=jnp.ones(1), usable=jnp.ones(1, dtype=bool))
	)

	return float(
		marginal.log_marginal(tau, quadrature.gauss_hermite(n_nodes), split, center=center)[0]
	)
