"""Item coordinate tables."""

from __future__ import annotations

import numpy as np


def initial_z(response: np.ndarray, n_items: int) -> np.ndarray:
	"""Starting values for the free item table, shape (n_items, Z_DIM)."""
	n = response.size
	if n == 0:
		raise ValueError("cannot initialize from an empty response array")

	interior = response[(response > 0.0) & (response < 1.0)]
	mean = float(interior.mean())
	var = float(interior.var())

	eta = float(np.log(mean / (1.0 - mean)))
	phi = mean * (1.0 - mean) / var - 1.0
	if phi <= 0.0:
		raise ValueError(f"interior responses are overdispersed for a Beta: phi={phi}")
	omega = 2.0 * float(np.log(phi / (2.0 * np.cosh(eta / 2.0))))

	# An empty boundary gets half an observation.
	floor = 0.5 / n
	p_zero = max(float((response == 0.0).mean()), floor)
	p_one = max(float((response == 1.0).mean()), floor)
	gamma_0 = float(np.log(p_zero / (1.0 - p_zero)))
	gamma_1 = float(np.log((1.0 - p_one) / p_one))
	if gamma_1 <= gamma_0:
		raise ValueError(f"boundary masses give gamma_1 <= gamma_0: {gamma_1} <= {gamma_0}")

	start = np.array([0.0, eta, omega, gamma_0, np.log(gamma_1 - gamma_0)])
	return np.tile(start, (n_items, 1))
