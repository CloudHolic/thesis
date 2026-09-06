"""Scale-linking diagnostic.

uv run scripts/02_linking_report.py (--response score)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import polars as pl

from thesis import config, domain, runmeta
from thesis.data import linking
from thesis.db import query


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--response", default=None, choices=sorted(domain.VIEWS))
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	args = ap.parse_args()

	cfg = config.load(args.config)
	response = args.response or cfg.data.response
	pool = args.pool or cfg.data.pool
	keys = cfg.data.keys

	started = time.monotonic()
	df = query.responses(dsn=cfg.db.dsn, response=response, keys=keys, pool=pool)
	query_seconds = time.monotonic() - started

	if df.height == 0:
		raise SystemExit(f"no rows for response={response} keys={list(keys)} pool={pool}")

	# An item is a (beatmap, rate_group) pair, not a beatmap.
	df = df.with_columns(
		(pl.col("beatmap_id").cast(pl.Utf8) + "|" + pl.col("rate_group")).alias("item")
	)

	present = sorted(int(k) for k in df["keys"].unique())
	table = linking.summary(df, keys=present, thresholds=cfg.data.diagnostics.link_thresholds)
	matrix = linking.pairwise(df, keys=present, threshold=cfg.data.link_threshold)
	report = linking.components(df)

	cfg.ensure_dirs()
	meta = runmeta.build(
		script="02_linking_report.py",
		config_raw=cfg.raw,
		extra={
			"effective": {
				"response": response,
				"pool": pool,
				"keys_requested": list(keys),
				"keys_present": present,
				"link_threshold": cfg.data.link_threshold,
			},
			"n_responses": df.height,
			"n_users": int(df["user_id"].n_unique()),
			"n_items": int(df["item"].n_unique()),
			"query_seconds": round(query_seconds, 1),
			"components": asdict(report),
		},
	)

	for name, frame in (("linking", table), ("linking_pairwise", matrix)):
		out = cfg.paths.artifacts / f"{name}_{response}_{pool}.parquet"
		frame.write_parquet(out)
		runmeta.write(out, meta)

	with pl.Config(tbl_rows=-1, tbl_cols=-1):
		print(table)
		print()
		print(f"pairwise co-play at threshold {cfg.data.link_threshold}")
		print(matrix)

	print(f"\ncomponents: {report.n_components:,}")
	print(f"  largest holds {report.largest_users:,} users, {report.largest_items:,} items")
	print(f"  and {report.largest_share_of_responses:.4%} of responses")
	print(f"  items by key inside : {report.items_by_key_in_largest}")
	print(f"  items by key outside: {report.items_by_key_outside}")
	print(f"\n{df.height:,} responses pulled in {query_seconds:.1f}s")


if __name__ == "__main__":
	main()
