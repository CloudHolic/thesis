"""Reading difficulty out of tau. The axes the pilot reports, and nothing else."""

from __future__ import annotations

import numpy as np

ACCURACIES: tuple[float, ...] = (0.9, 0.95, 0.98)


def tau_from_z(z: np.ndarray) -> np.ndarray:
	"""(a, b, omega, gamma_0, gamma_1) from the unconstrained coordinates."""
	return np.column_stack([np.exp(z[:, 0]), z[:, 1], z[:, 2], z[:, 3], z[:, 3] + np.exp(z[:, 4])])


def quantities(tau: np.ndarray) -> dict[str, np.ndarray]:
	"""What the pilot reports.

	The z coordinates are not it: lambda mixes a well-determined gamma_1 with an
	undetermined gamma_0, and reads as a failure when the information is fine.
	"""
	a = tau[:, 0]
	out = {f"b*({y:.2f})": (np.log(y / (1.0 - y)) - tau[:, 1]) / a for y in ACCURACIES}
	out["gamma_1/a"] = tau[:, 4] / a
	out["gamma_0/a"] = tau[:, 3] / a
	out["omega"] = tau[:, 2]

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
