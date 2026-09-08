"""The recovery table."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .reference import Posterior

ACCURACIES: tuple[float, ...] = (0.9, 0.95, 0.98)


def tau_from_z(z: np.ndarray) -> np.ndarray:
	"""(a, b, omega, gamma_0, gamma_1) from the unconstrained coordinates."""
	return np.column_stack([np.exp(z[:, 0]), z[:, 1], z[:, 2], z[:, 3], z[:, 3] + np.exp(z[:, 4])])


def tau_from_posterior(post: Posterior) -> np.ndarray:
	return np.column_stack(
		[
			np.exp(post.mean["log_a"]),
			post.mean["b"],
			post.mean["omega"],
			post.mean["gamma_0"],
			post.mean["gamma_1"],
		]
	)


def quantities(tau: np.ndarray) -> dict[str, np.ndarray]:
	"""What the model reports."""
	a = tau[:, 0]
	out = {f"b*({y:.2f})": (np.log(y / (1.0 - y)) - tau[:, 1]) / a for y in ACCURACIES}
	out["gamma_1/a"] = tau[:, 4] / a
	out["gamma_0/a"] = tau[:, 3] / a
	out["omega"] = tau[:, 2]
	return out


def compare(
	truth: dict[str, np.ndarray], estimate: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
	"""Bias, RMSE and rank correlation of `estimate` against `truth`, per quantity."""
	return {
		name: {
			"bias": float(np.mean(estimate[name] - truth[name])),
			"rmse": float(np.sqrt(np.mean((estimate[name] - truth[name]) ** 2))),
			"spearman": float(spearmanr(truth[name], estimate[name]).statistic),
		}
		for name in truth
	}


def render(table: dict[str, dict[str, float]], title: str) -> str:
	"""One block of the recovery table, for the run log."""
	lines = [f"{title:>12} {'bias':>9} {'rmse':>9} {'spearman':>9}"]
	lines += [
		f"{name:>12} {row['bias']:9.4f} {row['rmse']:9.4f} {row['spearman']:9.4f}"
		for name, row in table.items()
	]

	return "\n".join(lines)
