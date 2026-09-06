"""Whether the response graph puts every item on one scale.

theta ~ Normal(0, 1) standardizes each connected component of the user-item
graph independently, so item parameters from different components are not
comparable. Keys share no items and are the likeliest fault line, but the
constraint is connectivity, not the key count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def per_user_counts(df: pl.DataFrame, *, keys: Sequence[int]) -> pl.DataFrame:
	"""One row per user with an `n_<key>` column per key."""
	counts = df.group_by(["user_id", "keys"]).len(name="n")
	out = df.select("user_id").unique()

	for key in keys:
		per_key = counts.filter(pl.col("keys") == key).select(
			"user_id", pl.col("n").alias(f"n_{key}")
		)
		out = out.join(per_key, on="user_id", how="left")

	return out.with_columns(pl.col(f"n_{k}").fill_null(0) for k in keys).select(
		"user_id", *(f"n_{k}" for k in keys)
	)


def summary(df: pl.DataFrame, *, keys: Sequence[int], thresholds: Sequence[int]) -> pl.DataFrame:
	"""Per threshold: users meeting it in every key, and in each one alone.

	`n_both` collapses towards zero once more than two keys are listed, since
	almost nobody plays them all; `pairwise` is the readable form in that case.
	"""
	counts = per_user_counts(df, keys=keys)
	rows = []

	for threshold in thresholds:
		meets = {k: (pl.col(f"n_{k}") >= threshold) for k in keys}
		rows.append(
			{
				"threshold": threshold,
				"n_both": int(counts.filter(pl.all_horizontal(*meets.values())).height),
				**{f"n_key_{k}": int(counts.filter(meets[k]).height) for k in keys},
			}
		)

	return pl.DataFrame(rows)


def pairwise(df: pl.DataFrame, *, keys: Sequence[int], threshold: int) -> pl.DataFrame:
	"""Users meeting `threshold` in both keys, for every pair."""
	counts = per_user_counts(df, keys=keys)
	rows = []

	for left in keys:
		row: dict[str, int] = {"key": left}
		for right in keys:
			both = (pl.col(f"n_{left}") >= threshold) & (pl.col(f"n_{right}") >= threshold)
			row[f"n_{right}"] = int(counts.filter(both).height)

		rows.append(row)

	return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class ComponentReport:
	n_components: int
	n_users: int
	n_items: int
	largest_users: int
	largest_items: int
	largest_share_of_responses: float
	items_by_key_in_largest: dict[int, int]
	items_by_key_outside: dict[int, int]


def components(df: pl.DataFrame, *, item_col: str = "item") -> ComponentReport:
	"""Connected components of the user-item bipartite graph."""
	users = df.select("user_id").unique().with_row_index("u")
	items = df.select(item_col, "keys").unique(subset=item_col).with_row_index("i")
	edges = df.join(users, on="user_id").join(items.drop("keys"), on=item_col)

	n_users, n_items = users.height, items.height
	rows = edges["u"].to_numpy()
	cols = edges["i"].to_numpy() + n_users
	both_ways = (
		np.concatenate([rows, cols]),
		np.concatenate([cols, rows]),
	)
	adjacency = coo_matrix(
		(np.ones(len(rows) * 2, dtype=np.int8), both_ways),
		shape=(n_users + n_items, n_users + n_items),
	).tocsr()

	count, labels = connected_components(adjacency, directed=False)
	sizes = np.bincount(labels)
	largest = int(sizes.argmax())

	item_labels = labels[n_users:]
	in_largest = items.with_columns(pl.Series("in_largest", item_labels == largest))
	by_key = in_largest.group_by(["keys", "in_largest"]).len(name="n")

	def counted(flag: bool) -> dict[int, int]:
		rows = by_key.filter(pl.col("in_largest") == flag)
		return {int(k): int(n) for k, n in zip(rows["keys"], rows["n"], strict=True)}

	user_in_largest = labels[:n_users] == largest
	response_in_largest = int(edges.filter(pl.Series(user_in_largest)[edges["u"]]).height)

	return ComponentReport(
		n_components=int(count),
		n_users=n_users,
		n_items=n_items,
		largest_users=int(user_in_largest.sum()),
		largest_items=int((item_labels == largest).sum()),
		largest_share_of_responses=response_in_largest / edges.height,
		items_by_key_in_largest=counted(True),
		items_by_key_outside=counted(False),
	)
