"""Person-major response data. One dataset, one file, every response variable on it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class Dataset:
	"""Responses grouped by person."""

	offsets: np.ndarray
	item_index: np.ndarray
	held_out: np.ndarray
	responses: dict[str, np.ndarray]
	item: np.ndarray
	beatmap_id: np.ndarray
	rate_group: np.ndarray
	keys: np.ndarray
	user_id: np.ndarray

	@property
	def n_persons(self) -> int:
		return len(self.offsets) - 1

	@property
	def n_items(self) -> int:
		return len(self.item)

	@property
	def n_obs(self) -> int:
		return len(self.item_index)

	def responses_per_person(self) -> np.ndarray:
		return np.diff(self.offsets)

	def person_index(self) -> np.ndarray:
		"""The owning person of each response."""
		return np.repeat(np.arange(self.n_persons), self.responses_per_person())

	def items_table(self) -> pl.DataFrame:
		"""The item table, for the chart download list and for reports."""
		return pl.DataFrame(
			{
				"item_index": np.arange(self.n_items),
				"item": self.item,
				"beatmap_id": self.beatmap_id,
				"rate_group": self.rate_group,
				"keys": self.keys,
			}
		)


def build(df: pl.DataFrame, responses: Sequence[str], *, item_col: str = "item") -> Dataset:
	"""Group a response frame by person. `responses` names the response columns."""
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
		held_out=np.zeros(joined.height, dtype=bool),
		responses={name: joined[name].to_numpy().astype(np.float64) for name in responses},
		# polars hands strings back as object arrays, which npz can only store pickled.
		item=items[item_col].to_numpy().astype(np.str_),
		beatmap_id=items["beatmap_id"].to_numpy(),
		rate_group=items["rate_group"].to_numpy().astype(np.str_),
		keys=items["keys"].to_numpy(),
		user_id=users["user_id"].to_numpy(),
	)


def with_holdout(dataset: Dataset, held_out: np.ndarray) -> Dataset:
	"""The same dataset with its holdout mask set."""
	if held_out.shape != (dataset.n_obs,):
		raise ValueError(f"held_out must be ({dataset.n_obs},), got {held_out.shape}")

	return replace(dataset, held_out=held_out.astype(bool))


def save(dataset: Dataset, path: Path) -> None:
	"""Write the whole dataset to one npz. Response names are stored, not encoded in keys."""
	names = sorted(dataset.responses)
	np.savez_compressed(
		path,
		offsets=dataset.offsets,
		item_index=dataset.item_index,
		held_out=dataset.held_out,
		response_names=np.array(names),
		response=np.stack([dataset.responses[name] for name in names]),
		item=dataset.item,
		beatmap_id=dataset.beatmap_id,
		rate_group=dataset.rate_group,
		keys=dataset.keys,
		user_id=dataset.user_id,
	)


def load(path: Path) -> Dataset:
	"""Read back what `save` wrote."""
	arrays = np.load(path)
	names = [str(name) for name in arrays["response_names"]]

	return Dataset(
		offsets=arrays["offsets"],
		item_index=arrays["item_index"],
		held_out=arrays["held_out"],
		responses=dict(zip(names, arrays["response"], strict=True)),
		item=arrays["item"],
		beatmap_id=arrays["beatmap_id"],
		rate_group=arrays["rate_group"],
		keys=arrays["keys"],
		user_id=arrays["user_id"],
	)
