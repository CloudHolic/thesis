"""Apply the schema, then load osu! dump archives into PostgreSQL.

uv run scripts/00_database.py --schema-only
uv run scripts/00_database.py --dump-dir D:/dumps (--dry-run)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from thesis import config, db


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--schema", default="public", help="target schema")
	ap.add_argument("--dump-dir", type=Path, default=None, help="folder of *.tar.bz2 dumps")
	ap.add_argument("--schema-only", action="store_true", help="apply DDL and stop")
	ap.add_argument(
		"--dry-run",
		action="store_true",
		help="list the archives and their inferred pool flags, then stop",
	)
	args = ap.parse_args()

	if not args.schema_only and args.dump_dir is None:
		raise SystemExit("pass --dump-dir, or --schema-only to apply the DDL alone")

	cfg = config.load(args.config)

	if not args.dry_run:
		with psycopg.connect(cfg.db.dsn) as conn:
			db.apply_all(conn, schema=args.schema)
		print(f"applied schema.sql and views.sql to {args.schema!r}")
		if args.schema_only:
			return

	# Sorted so the newest snapshot is loaded last and wins the upsert: beatmap play
	# counts and star ratings change between dumps.
	paths = sorted(args.dump_dir.rglob("*.tar.bz2"))
	if not paths:
		raise SystemExit(f"no archives under {args.dump_dir}")

	for path in paths:
		in_top, in_random = db.pool_flags(path)
		print(f"{path.name}: in_top={in_top} in_random={in_random}")

	if args.dry_run:
		return

	if not any(any(db.pool_flags(p)) for p in paths):
		raise SystemExit(
			"no archive filename contains 'top' or 'random'; every row would carry both "
			"pool flags false and the random-pool filter would silently select nothing"
		)

	with psycopg.connect(cfg.db.dsn) as conn:
		for path in paths:
			print(f"\n{path.name}")
			results = db.load_dump(conn, path)
			db.record(conn, results)
			for r in results:
				print(
					f"  {r.member}: read {r.rows_read:,} "
					f"-> copied {r.rows_copied:,} -> written {r.rows_written:,}"
				)


if __name__ == "__main__":
	main()
