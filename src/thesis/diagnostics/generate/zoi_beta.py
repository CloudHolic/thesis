"""Synthetic responses from the ZOI Beta-IRT model, built without the fitting code."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from thesis.diagnostics.cells import Fixture

Z_DIM = 5

# True item parameters for the synthetic fixture, uniform in tau. Moment-matched to the
# observed acc marginal: E[y] = sigmoid(b), phi = 2 exp(omega/2) cosh(eta/2),
# P(y=1) = 1 - sigmoid(gamma_1). gamma_0 sits where the data cannot speak.
TAU_RANGES = np.array([[0.5, 2.5], [1.5, 5.0], [0.5, 3.5], [-7.0, -5.0], [2.5, 4.5]])
SEED: int = 20260907


def draw_z(rng: np.random.Generator, n_items: int, ranges: np.ndarray) -> np.ndarray:
	"""Uniform over the configured coordinate ranges, one row per item."""
	if ranges.shape != (Z_DIM, 2):
		raise ValueError(f"Ranges must be {(Z_DIM, 2)}, got {ranges.shape}")

	tau = rng.uniform(ranges[:, 0], ranges[:, 1], size=(n_items, Z_DIM))
	if (tau[:, 0] <= 0.0).any():
		raise ValueError("a must be positive")
	if (tau[:, 4] <= tau[:, 3]).any():
		raise ValueError("gamma_1 must exceed gamma_0")

	return np.column_stack(
		[np.log(tau[:, 0]), tau[:, 1], tau[:, 2], tau[:, 3], np.log(tau[:, 4] - tau[:, 3])]
	)


def draw_responses(
	rng: np.random.Generator,
	z: np.ndarray,
	theta: np.ndarray,
	item_index: np.ndarray,
	person_index: np.ndarray,
) -> np.ndarray:
	"""Responses from the exact mixture, using only numpy and scipy."""
	a = np.exp(z[item_index, 0])
	b = z[item_index, 1]
	omega = z[item_index, 2]
	gamma_0 = z[item_index, 3]
	gamma_1 = gamma_0 + np.exp(z[item_index, 4])

	a_theta = a * theta[person_index]
	eta = a_theta + b
	below = expit(gamma_0 - a_theta)
	upto = expit(gamma_1 - a_theta)

	u = rng.random(item_index.size)
	interior = (u >= below) & (u < upto)
	response = np.where(u < below, 0.0, 1.0)
	response[interior] = rng.beta(
		np.exp((eta[interior] + omega[interior]) / 2.0),
		np.exp((-eta[interior] + omega[interior]) / 2.0),
	)

	return response


def build(
	item_index: np.ndarray,
	person_index: np.ndarray,
	n_items: int,
	n_persons: int,
	ranges: np.ndarray,
	seed: int,
) -> Fixture:
	"""Draws truth and responses for a given set of cells."""
	rng = np.random.default_rng(seed)
	z = draw_z(rng, n_items, ranges)
	theta = rng.standard_normal(n_persons)

	return Fixture(
		item_index=item_index,
		person_index=person_index,
		response=draw_responses(rng, z, theta, item_index, person_index),
		z=z,
		theta=theta,
		n_items=n_items,
		n_persons=n_persons,
	)
