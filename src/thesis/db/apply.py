"""Idempotent application of the schema and view definitions."""

from __future__ import annotations

from importlib import resources
from typing import LiteralString, cast

import psycopg
from psycopg import sql

_FILES = ("schema.sql", "views.sql")


def read_sql(name: str) -> sql.SQL:
	"""One bundled .sql file, as a statement."""
	text = resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
	return sql.SQL(cast(LiteralString, text))


def apply_all(conn: psycopg.Connection, *, schema: str = "public") -> None:
	"""Apply schema.sql then views.sql inside `schema`."""
	with conn.cursor() as cur:
		cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
		for name in _FILES:
			cur.execute(read_sql(name))

	conn.commit()
