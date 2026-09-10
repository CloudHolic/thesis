"""Which (person, item) cells a synthetic fixture covers, and what a fixture carries."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from thesis.data import Dataset


class Fixture(NamedTuple):
	"""Synthetic responses and the truth that produced them."""

	item_index: np.ndarray
	person_index: np.ndarray
	response: np.ndarray
	z: np.ndarray
	theta: np.ndarray
	n_items: int
	n_persons: int


def full_cross(n_items: int, n_persons: int) -> tuple[np.ndarray, np.ndarray]:
	"""Person-sorted indices for every person answering every item."""
	return np.tile(np.arange(n_items), n_persons), np.repeat(np.arange(n_persons), n_items)


def training_cells(data: Dataset) -> tuple[np.ndarray, np.ndarray]:
	"""The cells a fit train on, in person order, so throughput carries over to fitting."""
	keep = ~data.held_out
	return data.item_index[keep], data.person_index()[keep]
