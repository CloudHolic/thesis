"""Reading difficulty out of tau. The axes the pilot reports, and nothing else."""

from __future__ import annotations

import numpy as np

from thesis.model.family.protocol import Family

ACCURACIES: tuple[float, ...] = (0.9, 0.95, 0.98)


def quantities(family: Family, tau: np.ndarray) -> dict[str, np.ndarray]:
	"""What the pilot reports."""
	out = {f"b*({y:.2f})": np.asarray(family.b_star(tau, y)) for y in ACCURACIES}
	out.update(family.quantities(tau))

	return out


def responder_range(
	item_index: np.ndarray, person_index: np.ndarray, theta: np.ndarray, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
	"""The theta span of the people who answered each item."""
	low = np.full(n_items, np.inf)
	high = np.full(n_items, -np.inf)
	np.minimum.at(low, item_index, theta[person_index])
	np.maximum.at(high, item_index, theta[person_index])

	return low, high


def extrapolated(
	quantities: dict[str, np.ndarray], low: np.ndarray, high: np.ndarray
) -> dict[str, np.ndarray]:
	"""Which thresholds fall outside the span they were measured on.

	A b*(y) beyond the ability of everyone who played the item is an extrapolation of
	the fitted curve, not something the responses witnessed.
	"""
	return {
		name: (values < low) | (values > high)
		for name, values in quantities.items()
		if name.startswith("b*")
	}
