"""Bidirectional k-core sweep.

uv run scripts/03_kcore_sweep.py (--response score)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from thesis import config, domain, runmeta
from thesis.data import kcore
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
	diagnostics = cfg.data.diagnostics

	started = time.monotonic()
	df = query.responses(dsn=cfg.db.dsn, response=response, keys=cfg.data.keys, pool=pool)
	query_seconds = time.monotonic() - started

	if df.height == 0:
		raise SystemExit(f"no rows for response={response} pool={pool}")

	# An item is a (beatmap, rate_group) pair, not a beatmap.
	df = df.with_columns(
		(pl.col("beatmap_id").cast(pl.Utf8) + "|" + pl.col("rate_group")).alias("item")
	)

	table = kcore.sweep(
		df, min_items=diagnostics.kcore_min_items, min_users=diagnostics.kcore_min_users
	)

	cfg.ensure_dirs()
	out = cfg.paths.artifacts / f"kcore_sweep_{response}_{pool}.parquet"
	table.write_parquet(out)
	runmeta.write(
		out,
		runmeta.build(
			script="03_kcore_sweep.py",
			config_raw=cfg.raw,
			extra={
				"effective": {
					"response": response,
					"pool": pool,
					"keys": list(cfg.data.keys),
					"min_items": list(diagnostics.kcore_min_items),
					"min_users": list(diagnostics.kcore_min_users),
				},
				"n_responses_before_filtering": df.height,
				"query_seconds": round(query_seconds, 1),
			},
		),
	)

	with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=200):
		print(table.drop("items_by_key"))

	print(f"\n{df.height:,} responses pulled in {query_seconds:.1f}s -> {out}")


if __name__ == "__main__":
	main()
