"""Builds the dataset: core, stratified sample, person-major, holdout."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from thesis import config, domain, runmeta
from thesis.data import dataset, holdout, kcore, linking, sample
from thesis.db import query


def response_shape(values: np.ndarray) -> dict[str, float]:
	"""Boundary masses and quantiles, which decide what the inflation terms can learn."""
	quantiles = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
	return {
		"at_zero": float((values == 0.0).mean()),
		"at_one": float((values == 1.0).mean()),
		"min": float(values.min()),
		"p01": float(quantiles[0]),
		"p25": float(quantiles[1]),
		"p50": float(quantiles[2]),
		"p75": float(quantiles[3]),
		"p99": float(quantiles[4]),
		"max": float(values.max()),
	}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	args = ap.parse_args()

	cfg = config.load(args.config)
	pool = args.pool or cfg.data.pool
	names = sorted(domain.VIEWS)
	first, *rest = names

	started = time.monotonic()
	frames = {
		name: query.responses(
			dsn=cfg.db.dsn, response=name, keys=cfg.data.keys, pool=pool
		).with_columns(
			# An item is a (beatmap, rate_group) pair, not a beatmap.
			(pl.col("beatmap_id").cast(pl.Utf8) + "|" + pl.col("rate_group")).alias("item")
		)
		for name in names
	}
	query_seconds = time.monotonic() - started
	for name, frame in frames.items():
		if frame.height == 0:
			raise SystemExit(f"no rows for response={name} pool={pool}")

	cells = {
		name: frame.select("user_id", "item").sort("user_id", "item")
		for name, frame in frames.items()
	}
	for name in rest:
		if not cells[first].equals(cells[name]):
			raise SystemExit(f"[{first}] and [{name}] do not cover the same cells")

	core = kcore.filter_kcore(
		frames[first], min_item=cfg.data.core.min_items, min_user=cfg.data.core.min_users
	)
	allocation = sample.allocate(
		sample.item_counts_by_key(core),
		n_items=cfg.data.sample.n_items,
		floor=cfg.data.sample.min_items_per_key,
	)
	drawn = sample.sample_items(
		core,
		n_items=cfg.data.sample.n_items,
		seed=cfg.data.sample.seed,
		floor=cfg.data.sample.min_items_per_key,
	).filter(pl.len().over("user_id") >= cfg.data.core.min_users)

	cell_keys = drawn.select("user_id", "item", "beatmap_id", "rate_group", "keys")
	built = {}
	for name in names:
		frame = cell_keys.join(
			frames[name].select("user_id", "item", "response"), on=["user_id", "item"]
		)
		if frame.height != drawn.height:
			raise SystemExit(f"[{name}] does not cover every drawn cell exactly once")
		built[name] = dataset.build(frame)

	data = built[first]
	for name in rest:
		if not np.array_equal(data.offsets, built[name].offsets) or not np.array_equal(
			data.item_index, built[name].item_index
		):
			raise SystemExit(f"[{name}] built a different layout than [{first}]")

	held_out = holdout.cell_mask(
		data.item_index,
		data.person_index(),
		fraction=cfg.data.holdout.cell_fraction,
		seed=cfg.data.holdout.seed,
		min_remaining=cfg.data.holdout.min_remaining,
	)

	per_person = data.responses_per_person()
	per_item = np.bincount(data.item_index, minlength=data.n_items)
	connectivity = linking.components(drawn)

	def spread(values: np.ndarray) -> dict[str, float]:
		return {
			"min": int(values.min()),
			"median": float(np.median(values)),
			"max": int(values.max()),
		}

	summary = {
		"n_items": data.n_items,
		"n_persons": data.n_persons,
		"n_obs": data.n_obs,
		"held_out": int(held_out.sum()),
		"held_out_requested": int(round(data.n_obs * cfg.data.holdout.cell_fraction)),
	}
	item_spread = spread(per_item)
	person_spread = spread(per_person)
	shapes = {name: response_shape(built[name].response) for name in names}

	print(f"core     {core['item'].n_unique():,} items, {core.height:,} responses")
	print(f"draw     {allocation}")
	print(
		f"dataset  {summary['n_items']:,} items, {summary['n_persons']:,} persons, "
		f"{summary['n_obs']:,} responses, shared by {', '.join(names)}"
	)
	print(f"held out {summary['held_out']:,} of {summary['held_out_requested']:,} requested")
	print(f"per item    {item_spread}")
	print(f"per person  {person_spread}")
	print(f"components  {connectivity.n_components}")

	cfg.ensure_dirs()
	for name in names:
		out = cfg.paths.artifacts / f"pilot_{name}_{pool}.npz"
		dataset.save(built[name], held_out, out)
		runmeta.write(
			out,
			runmeta.build(
				script="04_build_dataset.py",
				config_raw=cfg.raw,
				extra={
					"effective": {"response": name, "pool": pool, "shares_cells_with": names},
					"query_seconds": round(query_seconds, 1),
					"core": {"items": int(core["item"].n_unique()), "obs": core.height},
					"allocation": {str(k): v for k, v in allocation.items()},
					"dataset": summary,
					"responses_per_item": item_spread,
					"responses_per_person": person_spread,
					"response_shape": shapes[name],
					"components": {
						"n_components": connectivity.n_components,
						"largest_share_of_responses": connectivity.largest_share_of_responses,
						"items_by_key_outside": connectivity.items_by_key_outside,
					},
				},
			),
		)
		print(f"response {name:>5}  {shapes[name]}")
		print(f"wrote {out}")


if __name__ == "__main__":
	main()
