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
