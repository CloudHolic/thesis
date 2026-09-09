"""Diagnose the response graph, then build the pilot dataset.

uv run scripts/01_dataset.py                  diagnostics and build
uv run scripts/01_dataset.py --diagnose-only  linking and k-core sweep alone
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from thesis import config, db, domain, runmeta
from thesis.data import dataset, holdout, kcore, linking, sample

# Grids for the finished sweeps. The thresholds they justified are in decisions.md;
# these stay so the diagnostic artifacts can be reproduced.
LINK_THRESHOLDS = (1, 2, 5, 10, 20, 50)
LINK_THRESHOLD = 10
KCORE_MIN_ITEMS = (5, 10, 20, 50, 100)
KCORE_MIN_USERS = (5, 10, 20, 50)


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


def spread(values: np.ndarray) -> dict[str, float]:
	return {
		"min": int(values.min()),
		"median": float(np.median(values)),
		"max": int(values.max()),
	}


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	ap.add_argument("--diagnose-only", action="store_true", help="skip the dataset build")
	args = ap.parse_args()

	cfg = config.load(args.config)
	pool = args.pool or cfg.data.pool
	names = sorted(domain.VIEWS)

	started = time.monotonic()
	frames = {
		name: db.responses(
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

	# One frame with every response variable side by side. The join is what guarantees
	# the variables sit on the same cells; a height mismatch means they do not.
	merged = frames[names[0]].rename({"response": names[0]})
	for name in names[1:]:
		merged = merged.join(
			frames[name].select("user_id", "item", pl.col("response").alias(name)),
			on=["user_id", "item"],
		)
		if merged.height != frames[name].height:
			raise SystemExit(f"[{names[0]}] and [{name}] do not cover the same cells")

	base = frames[names[0]]
	diagnostics = cfg.directory("diagnostics")
	present = sorted(int(k) for k in base["keys"].unique())
	link_table = linking.summary(base, keys=present, thresholds=LINK_THRESHOLDS)
	link_matrix = linking.pairwise(base, keys=present, threshold=LINK_THRESHOLD)
	graph = linking.components(base)
	sweep = kcore.sweep(base, min_items=KCORE_MIN_ITEMS, min_users=KCORE_MIN_USERS)

	meta = runmeta.build(
		script="01_dataset.py",
		config_raw=cfg.raw,
		extra={
			"effective": {
				"pool": pool,
				"keys_requested": list(cfg.data.keys),
				"keys_present": present,
				"link_thresholds": list(LINK_THRESHOLDS),
				"link_threshold": LINK_THRESHOLD,
				"kcore_min_items": list(KCORE_MIN_ITEMS),
				"kcore_min_users": list(KCORE_MIN_USERS),
			},
			"n_responses": base.height,
			"n_users": int(base["user_id"].n_unique()),
			"n_items": int(base["item"].n_unique()),
			"query_seconds": round(query_seconds, 1),
			"components": asdict(graph),
		},
	)
	for name, frame in (
		("linking", link_table),
		("linking_pairwise", link_matrix),
		("kcore_sweep", sweep),
	):
		out = diagnostics / f"{name}_{pool}.parquet"
		frame.write_parquet(out)
		runmeta.write(out, meta)

	with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=200):
		print(link_table)
		print(f"\npairwise co-play at threshold {LINK_THRESHOLD}")
		print(link_matrix)
		print()
		print(sweep.drop("items_by_key"))
	print(
		f"\ncomponents  {graph.n_components:,}, largest holds "
		f"{graph.largest_share_of_responses:.4%} of responses"
	)
	print(f"pulled      {base.height:,} responses in {query_seconds:.1f}s")

	if args.diagnose_only:
		return

	core = kcore.filter_kcore(
		merged,
		min_item=cfg.data.min_responses_per_item,
		min_user=cfg.data.min_responses_per_user,
	)
	allocation = sample.allocate(
		sample.item_counts_by_key(core),
		n_items=cfg.data.n_items,
		floor=cfg.data.min_items_per_key,
	)
	drawn = sample.sample_items(
		core,
		n_items=cfg.data.n_items,
		seed=cfg.data.sample_seed,
		floor=cfg.data.min_items_per_key,
	).filter(pl.len().over("user_id") >= cfg.data.min_responses_per_user)

	data = dataset.build(drawn, names)
	data = dataset.with_holdout(
		data,
		holdout.cell_mask(
			data.item_index,
			data.person_index(),
			fraction=cfg.data.holdout_fraction,
			seed=cfg.data.holdout_seed,
			min_remaining=cfg.data.holdout_min_remaining,
		),
	)

	per_item = np.bincount(data.item_index, minlength=data.n_items)
	per_person = data.responses_per_person()
	connectivity = linking.components(drawn)
	summary = {
		"n_items": data.n_items,
		"n_persons": data.n_persons,
		"n_obs": data.n_obs,
		"held_out": int(data.held_out.sum()),
		"held_out_requested": int(round(data.n_obs * cfg.data.holdout_fraction)),
	}
	shapes = {name: response_shape(values) for name, values in data.responses.items()}

	out = cfg.directory("dataset") / f"pilot_{pool}.npz"
	dataset.save(data, out)
	runmeta.write(
		out,
		runmeta.build(
			script="01_dataset.py",
			config_raw=cfg.raw,
			extra={
				"effective": {"pool": pool, "responses": names},
				"query_seconds": round(query_seconds, 1),
				"core": {"items": int(core["item"].n_unique()), "obs": core.height},
				"allocation": {str(k): v for k, v in allocation.items()},
				"dataset": summary,
				"responses_per_item": spread(per_item),
				"responses_per_person": spread(per_person),
				"response_shape": shapes,
				"components": {
					"n_components": connectivity.n_components,
					"largest_share_of_responses": connectivity.largest_share_of_responses,
					"items_by_key_outside": connectivity.items_by_key_outside,
				},
			},
		),
	)

	print(f"\ncore        {core['item'].n_unique():,} items, {core.height:,} responses")
	print(f"draw        {allocation}")
	print(
		f"dataset     {summary['n_items']:,} items, {summary['n_persons']:,} persons, "
		f"{summary['n_obs']:,} responses, shared by {', '.join(names)}"
	)
	print(f"held out    {summary['held_out']:,} of {summary['held_out_requested']:,} requested")
	print(f"per item    {spread(per_item)}")
	print(f"per person  {spread(per_person)}")
	for name, shape in shapes.items():
		print(f"response {name:>5}  {shape}")
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()
