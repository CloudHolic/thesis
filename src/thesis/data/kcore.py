"""Bidirectional iterative k-core filtering and threshold sweeps."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from . import linking


def filter_kcore(
	df: pl.DataFrame, *, min_item: int, min_user: int, item_col: str = "item"
) -> pl.DataFrame:
	"""Rows surviving the joint (item >= min_item, person >= min_user) core."""
	while True:
		before = df.height
		df = df.filter(
			(pl.len().over(item_col) >= min_item) & (pl.len().over("user_id") >= min_user)
		)
		if df.height == before:
			return df


def _rate_group_pairs(df: pl.DataFrame) -> int:
	"""Beatmaps surviving in more than one rate group."""
	if "beatmap_id" not in df.columns or "rate_group" not in df.columns:
		return 0

	per_map = df.group_by("beatmap_id").agg(pl.col("rate_group").n_unique().alias("n"))
	return int(per_map.filter(pl.col("n") > 1).height)


def _items_by_key(df: pl.DataFrame, item_col: str) -> dict[int, int]:
	"""Surviving item count per key."""
	if "keys" not in df.columns:
		return {}

	per_key = df.group_by("keys").agg(pl.col(item_col).n_unique().alias("n")).sort("keys")
	return {int(k): int(n) for k, n in zip(per_key["keys"], per_key["n"], strict=True)}


def sweep(
	df: pl.DataFrame, *, item_col: str = "item", min_items: Sequence[int], min_users: Sequence[int]
) -> pl.DataFrame:
	"""One row per (min_item, min_user) with the trade-offs the decision needs."""
	rows: list[dict[str, Any]] = []

	for min_item in min_items:
		for min_user in min_users:
			kept = filter_kcore(df, min_item=min_item, min_user=min_user, item_col=item_col)

			if kept.height == 0:
				rows.append(
					{
						"min_item": min_item,
						"min_user": min_user,
						"n_items": 0,
						"n_persons": 0,
						"n_obs": 0,
						"median_resp_per_item": 0.0,
						"n_rate_group_pairs": 0,
						"n_components": 0,
						"largest_share": 0.0,
						"n_keys": 0,
						"items_by_key": "{}",
					}
				)

				continue

			per_item = kept.group_by(item_col).len(name="n")
			median = per_item["n"].median()
			median_response = float(median) if isinstance(median, int | float) else 0.0
			report = linking.components(kept, item_col=item_col)
			by_key = _items_by_key(kept, item_col)

			rows.append(
				{
					"min_item": min_item,
					"min_user": min_user,
					"n_items": int(kept[item_col].n_unique()),
					"n_persons": int(kept["user_id"].n_unique()),
					"n_obs": int(kept.height),
					"median_resp_per_item": median_response,
					"n_rate_group_pairs": _rate_group_pairs(kept),
					"n_components": report.n_components,
					"largest_share": report.largest_share_of_responses,
					"n_keys": len(by_key),
					"items_by_key": str(by_key),
				}
			)

	return pl.DataFrame(rows)
