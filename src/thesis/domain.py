"""Facts already fixed, not configurable."""

from __future__ import annotations

RATE_GROUP_CLOCK: dict[str, float] = {"NM": 1.0, "DT": 1.5, "HT": 0.75}
VIEWS: dict[str, str] = {"acc": "v_response_acc", "score": "v_response_score"}
POOLS: tuple[str, ...] = ("random", "top", "all")
