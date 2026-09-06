"""Drawing a pilot item set from the core."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl


class AllocationError(ValueError):
    """The requested sample cannot honor the per-key floor."""


def allocate(available: Mapping[int, int], *, n_items: int, floor: int = 1) -> dict[int, int]:
    """How many items to draw from each key."""
    present = {k: n for k, n in available.items() if n > 0}
    if not present:
        raise AllocationError("no key has any item available")

    needed = sum(min(floor, n) for n in present.values())
    if n_items < needed:
        raise AllocationError(
            f"n_items={n_items} cannot give {floor} item(s) to each of "
            f"{len(present)} keys (needs at least {needed})"
        )
    if n_items > sum(present.values()):
        raise AllocationError(f"n_items={n_items} exceeds the {sum(present.values())} items available")

    total = sum(present.values())
    quota = {k: n / total * n_items for k, n in present.items()}
    alloc = {k: min(present[k], max(min(floor, present[k]), int(q))) for k, q in quota.items()}

    remainder = {k: quota[k] - int(quota[k]) for k in present}
    short = n_items - sum(alloc.values())

    while short > 0:
        room = [k for k in present if alloc[k] < present[k]]
        if not room:
            break

        pick = max(room, key=lambda k: (remainder[k], present[k], k))
        alloc[pick] += 1
        remainder[pick] = 0.0
        short -= 1

    while short < 0:
        room = [k for k in present if alloc[k] > min(floor, present[k])]
        if not room:
            break

        pick = max(room, key=lambda k: (alloc[k], k))
        alloc[pick] -= 1
        short += 1

    return dict(sorted(alloc.items()))


def item_counts_by_key(df: pl.DataFrame, *, item_col: str = "item") -> dict[int, int]:
    """Distinct items per key in a response frame."""
    per_key = df.group_by("keys").agg(pl.col(item_col).n_unique().alias("n")).sort("keys")
    return {int(k): int(n) for k, n in zip(per_key["keys"], per_key["n"], strict=True)}


def sample_items(df: pl.DataFrame, *, n_items: int, seed: int, item_col: str = "item", floor: int = 1) -> pl.DataFrame:
    """Every response on a stratified draw of `n_items` items."""
    allocation = allocate(item_counts_by_key(df, item_col=item_col), n_items=n_items, floor=floor)

    items = df.select(item_col, "keys").unique(subset=item_col)
    drawn = (
        items.sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(pl.int_range(pl.len()).over("keys").alias("rank"))
        .filter(pl.col("rank") < pl.col("keys").replace_strict(allocation, return_dtype=pl.Int64))
        .select(item_col)
    )

    return df.join(drawn, on=item_col, how="semi")