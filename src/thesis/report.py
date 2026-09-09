"""Turning results into markdown, so every artifact has a form a person can read."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

PERCENTILES = (5, 25, 50, 75, 95)


def table(header: Sequence[str], rows: Sequence[Sequence[str]], *, right: int = 1) -> str:
	"""A markdown table. Columns from `right` onward are right-aligned."""
	rule = ["---" if i < right else "---:" for i in range(len(header))]
	lines = [f"| {' | '.join(header)} |", f"|{'|'.join(rule)}|"]
	lines += [f"| {' | '.join(row)} |" for row in rows]

	return "\n".join(lines)


def comparison(block: dict[str, dict[str, float]], title: str) -> str:
	"""Bias, RMSE and rank correlation, one row per quantity."""
	return table(
		[title, "bias", "rmse", "spearman"],
		[
			[f"`{name}`", f"{row['bias']:.4f}", f"{row['rmse']:.4f}", f"{row['spearman']:.4f}"]
			for name, row in block.items()
		],
	)


def distribution(quantities: dict[str, np.ndarray]) -> str:
	"""Percentiles of each reported quantity, for a fit with no truth to compare to."""
	return table(
		["quantity", *(f"p{p:02d}" for p in PERCENTILES)],
		[
			[f"`{name}`", *(f"{v:.3f}" for v in np.percentile(values, PERCENTILES))]
			for name, values in quantities.items()
		],
	)


def shares(values: dict[str, float], header: Sequence[str]) -> str:
	"""One share per row, as a percentage."""
	return table(header, [[f"`{name}`", f"{share:.1%}"] for name, share in values.items()])
