"""What a likelihood family and its interior must provide."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from jax import Array


class Interior(Protocol):
	"""A density on (0, 1) with no atoms, and the coordinates we fit it in."""

	N_PARAMS: int
	TAU_NAMES: tuple[str, ...]
	SITES: tuple[str, ...]
	prior_scale: float

	def to_tau(self, z: Array, /) -> Any:
		"""Interior parameters from the unconstrained coordinates. Must expose `.a`."""
		...

	def tau_from_sites(self, means: dict[str, np.ndarray], /) -> Any:
		"""The same parameters from a reference posterior, which samples them directly."""
		...

	def interior_stats(self, y: np.ndarray, /) -> tuple[np.ndarray, ...]:
		"""What the density needs from the response and the caller can precompute once."""
		...

	def log_f(self, tau: Any, theta: Array, stats: tuple[Array, ...], /) -> Array:
		"""log of the density, on arguments already aligned for broadcasting."""
		...

	def initial_z(self, response: np.ndarray, /) -> np.ndarray:
		"""Starting coordinates for one item, shape (N_PARAMS,)."""
		...

	def b_star(self, tau: Any, f: float, /) -> Array:
		"""The ability at which this item is answered at y. Larger is harder."""
		...

	def quantities(self, tau: Any, /) -> dict[str, np.ndarray]:
		"""Reported derived values other than b*(y)."""
		...


@dataclass(frozen=True, slots=True)
class Family:
	"""A complete response distribution on [0, 1], atoms included."""

	Z_DIM: int
	TAU_NAMES: tuple[str, ...]
	SITES: tuple[str, ...]
	prior_scale: float

	to_tau: Callable[[Array], Any]
	tau_from_sites: Callable[[dict[str, np.ndarray]], Any]
	interior_stats: Callable[[np.ndarray], tuple[np.ndarray, ...]]
	log_k_zero: Callable[[Any, Array], Array]
	log_k_interior: Callable[[Any, Array, tuple[Array, ...]], Array]
	log_k_one: Callable[[Any, Array], Array]
	initial_z: Callable[[np.ndarray, int], np.ndarray]
	b_star: Callable[[Any, float], Array]
	quantities: Callable[[Any], dict[str, np.ndarray]]
