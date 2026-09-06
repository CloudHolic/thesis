"""Person-major response data and its holdout splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class Dataset:
	"""Responses grouped by person."""

	offsets: np.ndarray
	item_index: np.ndarray
	response: np.ndarray
	users: pl.DataFrame
	items: pl.DataFrame

	@property
	def n_persons(self) -> int:
		return len(self.offsets) - 1

	@property
	def n_items(self) -> int:
		return self.items.height

	@property
	def n_obs(self) -> int:
		return len(self.response)

	def responses_per_person(self) -> np.ndarray:
		return np.diff(self.offsets)

	def person_index(self) -> np.ndarray:
		"""The owning person of each response."""
		return np.repeat(np.arange(self.n_persons), self.responses_per_person())


def build(df: pl.DataFrame, *, item_col: str = "item") -> Dataset:
	"""Group a response frame by person."""
	users = df.select("user_id").unique().sort("user_id").with_row_index("person_index")
	items = (
		df.select(item_col, "beatmap_id", "rate_group", "keys")
		.unique(subset=item_col)
		.sort(item_col)
		.with_row_index("item_index")
	)

	joined = (
		df.join(users, on="user_id")
		.join(items.select(item_col, "item_index"), on=item_col)
		.sort("person_index", "item_index")
	)
	counts = joined.group_by("person_index").len(name="n").sort("person_index")["n"].to_numpy()

	return Dataset(
		offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
		item_index=joined["item_index"].to_numpy().astype(np.int64),
		response=joined["response"].to_numpy().astype(np.float64),
		users=users.select("user_id", "person_index"),
		items=items.select(item_col, "beatmap_id", "rate_group", "keys", "item_index"),
	)


def save(dataset: Dataset, held_out: np.ndarray, path: Path) -> None:
	"""Write the arrays to `path` and the tables beside it."""
	np.savez_compressed(
		path,
		offsets=dataset.offsets,
		item_index=dataset.item_index,
		response=dataset.response,
		held_out=held_out,
	)
	dataset.items.write_parquet(path.with_name(f"{path.stem}_items.parquet"))
	dataset.users.write_parquet(path.with_name(f"{path.stem}_users.parquet"))


def load(path: Path) -> tuple[Dataset, np.ndarray]:
	"""Read back what `save` wrote."""
	arrays = np.load(path)
	dataset = Dataset(
		offsets=arrays["offsets"],
		item_index=arrays["item_index"],
		response=arrays["response"],
		users=pl.read_parquet(path.with_name(f"{path.stem}_users.parquet")),
		items=pl.read_parquet(path.with_name(f"{path.stem}_items.parquet")),
	)

	return dataset, arrays["held_out"]
