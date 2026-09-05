"""Apply the schema and view definitions to the configured database.

uv run scripts/00_apply_schema.py (--schema staging)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from thesis import config
from thesis.db import apply


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--schema", default="public", help="target schema (default: public)")
	args = ap.parse_args()

	cfg = config.load(args.config)
	with psycopg.connect(cfg.db.dsn) as conn:
		apply.apply_all(conn, schema=args.schema)

	print(f"applied schema.sql and views.sql to {args.schema!r}")


if __name__ == "__main__":
	main()
