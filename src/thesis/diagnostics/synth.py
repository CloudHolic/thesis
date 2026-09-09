"""Synthetic responses from the ZOI Beta-IRT model, built without the fitting code."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import expit

from thesis.data import Dataset
from thesis.model.likelihood import Z_DIM


class Fixture(NamedTuple):
	"""Synthetic responses and the truth that produced them."""

	item_index: np.ndarray
	person_index: np.ndarray
	response: np.ndarray
	z: np.ndarray
	theta: np.ndarray
	n_items: int
	n_persons: int


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


def full_cross(n_items: int, n_persons: int) -> tuple[np.ndarray, np.ndarray]:
	"""Person-sorted indices for every person answering every item."""
	return np.tile(np.arange(n_items), n_persons), np.repeat(np.arange(n_persons), n_items)


def training_cells(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
	"""The cells a fit train on, in person order, so throughput carries over to fitting."""
	keep = ~data.held_out
	return data.item_index[keep], data.person_index()[keep]


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
