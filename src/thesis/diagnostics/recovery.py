"""Comparing two sets of item estimates on the axes the pilot reports."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from thesis.references.zoi import Posterior


def tau_from_posterior(post: Posterior) -> np.ndarray:
	"""(a, b, omega, gamma_0, gamma_1) from the reference posterior means."""
	return np.column_stack(
		[
			np.exp(post.mean["log_a"]),
			post.mean["b"],
			post.mean["omega"],
			post.mean["gamma_0"],
			post.mean["gamma_1"],
		]
	)


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
