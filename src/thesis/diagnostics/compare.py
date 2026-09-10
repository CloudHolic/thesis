"""Comparing two sets of item estimates on the axes the pilot reports."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


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
