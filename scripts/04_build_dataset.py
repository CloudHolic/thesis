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
        "max": float(values.max())
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
    ap.add_argument("--response", default=None, choices=sorted(domain.VIEWS))
    ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
    args = ap.parse_args()

    cfg = config.load(args.config)
    response = args.response or cfg.data.response
    pool = args.pool or cfg.data.pool

    started = time.monotonic()
    df = query.responses(dsn=cfg.db.dsn, response=response, keys=cfg.data.keys, pool=pool)
    query_seconds = time.monotonic() - started
    if df.height == 0:
        raise SystemExit(f"no rows for response={response} pool={pool}")

    # An item is a (beatmap, rate_group) pair, not a beatmap.
    df = df.with_columns(
        (pl.col("beatmap_id").cast(pl.Utf8) + "|" + pl.col("rate_group")).alias("item")
    )

    core = kcore.filter_kcore(
        df,
        min_item=cfg.data.core.min_items,
        min_user=cfg.data.core.min_users
    )
    allocation = sample.allocate(
        sample.item_counts_by_key(core),
        n_items=cfg.data.sample.n_items,
        floor=cfg.data.sample.min_items_per_key
    )
    drawn = sample.sample_items(
        core,
        n_items=cfg.data.sample.n_items,
        seed=cfg.data.sample.seed,
        floor=cfg.data.sample.min_items_per_key
    ).filter(pl.len().over("user_id") >= cfg.data.core.min_users)

    data = dataset.build(drawn)
    held_out = holdout.cell_mask(
        data.item_index,
        data.person_index(),
        fraction=cfg.data.holdout.cell_fraction,
        seed=cfg.data.holdout.seed,
        min_remaining=cfg.data.holdout.min_remaining
    )

    per_person = data.responses_per_person()
    per_item = np.bincount(data.item_index, minlength=data.n_items)
    connectivity = linking.components(drawn)

    def spread(values: np.ndarray) -> dict[str, float]:
        return {
            "min": int(values.min()),
            "median": float(np.median(values)),
            "max": int(values.max())
        }

    summary = {
        "n_items": data.n_items,
        "n_persons": data.n_persons,
        "n_obs": data.n_obs,
        "held_out": int(held_out.sum()),
        "held_out_requested": int(round(data.n_obs * cfg.data.holdout.cell_fraction))
    }
    item_spread = spread(per_item)
    person_spread = spread(per_person)
    shape = response_shape(data.response)

    cfg.ensure_dirs()
    out = cfg.paths.artifacts / f"pilot_{response}_{pool}.npz"
    dataset.save(data, held_out, out)
    runmeta.write(
        out,
        runmeta.build(
            script="04_build_dataset.py",
            config_raw=cfg.raw,
            extra={
                "effective": {"response": response, "pool": pool},
                "query_seconds": round(query_seconds, 1),
                "core": {"items": int(core["item"].n_unique()), "obs": core.height},
                "allocation": {str(k): v for k, v in allocation.items()},
                "dataset": summary,
                "responses_per_item": item_spread,
                "responses_per_person": person_spread,
                "response_shape": shape,
                "components": {
                    "n_components": connectivity.n_components,
                    "largest_share_of_responses": connectivity.largest_share_of_responses,
                    "items_by_key_outside": connectivity.items_by_key_outside
                }
            }
        )
    )

    print(f"core     {core['item'].n_unique():,} items, {core.height:,} responses")
    print(f"draw     {allocation}")
    print(f"dataset  {summary['n_items']:,} items, {summary['n_persons']:,} persons, "
          f"{summary['n_obs']:,} responses")
    print(f"held out {summary['held_out']:,} of {summary['held_out_requested']:,} requested")
    print(f"per item    {item_spread}")
    print(f"per person  {person_spread}")
    print(f"components  {connectivity.n_components}")
    print(f"response    {shape}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()