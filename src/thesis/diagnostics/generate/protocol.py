"""What a synthetic generator must provide."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from thesis.diagnostics.cells import Fixture


class Generator(Protocol):
	"""Draws truth and responses for one family, independently of the fitting code."""

	TAU_RANGES: np.ndarray
	SEED: int

	def draw_z(self, rng: np.random.Generator, n_items: int, ranges: np.ndarray, /) -> np.ndarray:
		"""Unconstrained item coordinates, uniform over `ranges` in tau space."""
		...

	def draw_responses(
		self,
		rng: np.random.Generator,
		z: np.ndarray,
		theta: np.ndarray,
		item_index: np.ndarray,
		person_index: np.ndarray,
		/,
	) -> np.ndarray:
		"""Responses from the exact mixture."""
		...

	def build(
		self,
		item_index: np.ndarray,
		person_index: np.ndarray,
		n_items: int,
		n_persons: int,
		ranges: np.ndarray,
		seed: int,
		/,
	) -> Fixture: ...
