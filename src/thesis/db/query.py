"""Read-only access to the response views."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .. import domain


def responses(*, dsn: str, response: str, keys: Sequence[int], pool: str) -> pl.DataFrame:
	"""(user_id, beatmap_id, rate_group, response, keys) for one pool."""
	if response not in domain.VIEWS:
		raise ValueError(f"response must be one of {sorted(domain.VIEWS)}")
	if pool not in domain.POOLS:
		raise ValueError(f"pool must be one of {list(domain.POOLS)}")

	key_list = ", ".join(str(int(k)) for k in keys)

	if pool not in domain.POOLS:
		raise ValueError(f"pool must be one of {list(domain.POOLS)}")

	where = [f"keys IN ({key_list})", "pool = %s"]

	sql = (
		f"SELECT user_id, beatmap_id, rate_group, response, keys "
		f"FROM {domain.VIEWS[response]} WHERE {' AND '.join(where)}"
	)
	return pl.read_database_uri(sql, dsn, engine="connectorx")
