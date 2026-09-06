"""Held-out cells."""

from __future__ import annotations

import numpy as np


def cell_mask(
    item_index: np.ndarray,
    person_index: np.ndarray,
    *,
    fraction: float,
    seed: int,
    min_remaining: int = 2
) -> np.ndarray:
    """Boolean mask over response, True where the response is held out."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if min_remaining < 1:
        raise ValueError("min_remaining must be at least 1")

    n_obs = len(item_index)
    items_left = np.bincount(item_index)
    persons_left = np.bincount(person_index)

    mask = np.zeros(n_obs, dtype=bool)
    target = int(round(n_obs * fraction))
    hidden = 0

    for cell in np.random.default_rng(seed).permutation(n_obs):
        if hidden >= target:
            break

        item, person = item_index[cell], person_index[cell]
        if items_left[item] <= min_remaining or persons_left[person] <= min_remaining:
            continue

        mask[cell] = True
        items_left[item] -= 1
        persons_left[person] -= 1
        hidden += 1

    return mask