from __future__ import annotations

RATE_GROUP_CLOCK: dict[str, float] = {"NM": 1.0, "DT": 1.5, "HT": 0.75}
VIEWS: dict[str, str] = {"acc": "v_irt_acc", "score": "v_irt_score"}
POOLS: tuple[str, ...] = {"random", "top", "all"}
