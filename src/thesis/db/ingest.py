"""Loading osu! dump archives into PostgreSQL."""

from __future__ import annotations

import tarfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from . import dump

SCORE_TABLE = "scores"
BEATMAP_TABLE = "beatmaps"

MEMBER_TABLES = {
	"osu_scores_mania_high": SCORE_TABLE,
	"osu_beatmaps": BEATMAP_TABLE,
}

_COLUMNS = {
	SCORE_TABLE: tuple(f.name for f in fields(dump.ScoreRow)),
	BEATMAP_TABLE: tuple(f.name for f in fields(dump.BeatmapRow)),
}
_PRIMARY_KEY = {SCORE_TABLE: "score_id", BEATMAP_TABLE: "beatmap_id"}

REQUIRED_COLUMNS = {
	SCORE_TABLE: {
		0: "score_id",
		1: "beatmap_id",
		2: "user_id",
		3: "score",
		5: "rank",
		6: "count50",
		7: "count100",
		8: "count300",
		9: "countmiss",
		10: "countgeki",
		11: "countkatu",
		13: "enabled_mods",
		14: "date",
		15: "pp",
	},
	BEATMAP_TABLE: {
		0: "beatmap_id",
		1: "beatmapset_id",
		5: "version",
		6: "total_length",
		13: "diff_size",
		16: "playmode",
		19: "difficultyrating",
		21: "playcount",
		22: "passcount",
		27: "bpm",
	},
}


class DumpSchemaError(RuntimeError):
	"""A dump member's columns are not where the parser expects them."""


@dataclass(frozen=True, slots=True)
class MemberResult:
	dump_file: str
	member: str
	target_table: str
	in_top: bool
	in_random: bool
	rows_read: int
	rows_copied: int
	rows_written: int
	started_at: datetime
	finished_at: datetime


def pool_flags(dump_path: Path) -> tuple[bool, bool]:
	"""(in_top, in_random) implied by the archive filename."""
	name = dump_path.name.lower()
	return "top" in name, "random" in name


def declared_columns(header: list[str]) -> list[str]:
	"""Column names from a mysqldump CREATE TABLE block, in order.

	Index and constraint lines start with a keyword rather than a backtick, so
	they are skipped.
	"""
	names: list[str] = []
	inside = False

	for line in header:
		if line.startswith("CREATE TABLE"):
			inside = True
			continue
		if not inside:
			continue

		stripped = line.strip()
		if stripped.startswith("`"):
			names.append(stripped.split("`")[1])
		elif stripped.startswith(")"):
			break

	return names


def verify_columns(member: str, table: str, header: list[str]) -> None:
	"""Raise unless every index the parser reads holds the column it expects."""
	declared = declared_columns(header)
	if not declared:
		raise DumpSchemaError(f"{member}: no CREATE TABLE block, cannot verify column order")

	wrong = {
		index: (expected, declared[index] if index < len(declared) else None)
		for index, expected in REQUIRED_COLUMNS[table].items()
		if index >= len(declared) or declared[index] != expected
	}
	if wrong:
		raise DumpSchemaError(
			f"{member}: dump column order changed, index -> (expected, declared): {wrong}"
		)


def _convert(table: str, values: list[Any], *, in_top: bool, in_random: bool):
	if table == SCORE_TABLE:
		return dump.to_score_row(values, in_top=in_top, in_random=in_random)
	return dump.to_beatmap_row(values)


def _identifiers(names: Sequence[str]) -> sql.Composed:
	return sql.SQL(", ").join(sql.Identifier(n) for n in names)


def _upsert_sql(table: str, staging: str) -> sql.Composed:
	columns = _COLUMNS[table]
	key = _PRIMARY_KEY[table]

	if table == SCORE_TABLE:
		update = sql.SQL(
			"in_top = {table}.in_top OR EXCLUDED.in_top, "
			"in_random = {table}.in_random OR EXCLUDED.in_random"
		).format(table=sql.Identifier(table))
	else:
		update = sql.SQL(", ").join(
			sql.SQL("{column} = EXCLUDED.{column}").format(column=sql.Identifier(c))
			for c in columns
			if c != key
		)

	return sql.SQL(
		"INSERT INTO {table} ({columns}) "
		"SELECT DISTINCT ON ({key}) {columns} FROM {staging} "
		"ON CONFLICT ({key}) DO UPDATE SET {update}"
	).format(
		table=sql.Identifier(table),
		columns=_identifiers(columns),
		key=sql.Identifier(key),
		staging=sql.Identifier(staging),
		update=update,
	)


def _split_header(handle: Iterator[bytes]) -> tuple[list[str], str | None]:
	"""Lines before the first INSERT, and that INSERT line itself."""
	header: list[str] = []

	for raw in handle:
		line = raw.decode("utf-8", errors="replace").rstrip()
		if line.startswith("INSERT INTO"):
			return header, line
		header.append(line)

	return header, None


def _lines(first: str | None, handle: Iterator[bytes]) -> Iterator[str]:
	if first is not None:
		yield first
	for raw in handle:
		yield raw.decode("utf-8", errors="replace").rstrip()


def load_member(
	conn: psycopg.Connection,
	handle: Iterator[bytes],
	table: str,
	*,
	member: str,
	in_top: bool,
	in_random: bool,
) -> tuple[int, int, int]:
	"""Verify the member's columns, COPY it into staging, then upsert."""
	header, first_insert = _split_header(handle)
	verify_columns(member, table, header)

	staging = f"stg_{table}"
	read = copied = 0

	with conn.cursor() as cur:
		cur.execute(
			sql.SQL(
				"CREATE TEMP TABLE {staging} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP"
			).format(staging=sql.Identifier(staging), table=sql.Identifier(table))
		)

		copy_statement = sql.SQL("COPY {staging} ({columns}) FROM STDIN").format(
			staging=sql.Identifier(staging), columns=_identifiers(_COLUMNS[table])
		)

		with cur.copy(copy_statement) as copy:
			for line in _lines(first_insert, handle):
				if MEMBER_TABLES.get(dump.table_of(line)) != table:
					continue

				for values in dump.parse_values_line(line):
					read += 1
					row = _convert(table, values, in_top=in_top, in_random=in_random)
					if row is None:
						continue
					copy.write_row(row.as_tuple())
					copied += 1

		cur.execute(_upsert_sql(table, staging))
		written = cur.rowcount

	conn.commit()
	return read, copied, written


def load_dump(conn: psycopg.Connection, dump_path: Path) -> list[MemberResult]:
	"""Load every recognized member of one archive."""
	in_top, in_random = pool_flags(dump_path)
	results: list[MemberResult] = []

	with tarfile.open(dump_path, "r:bz2") as tar:
		for info in tar.getmembers():
			if not info.name.endswith(".sql"):
				continue
			table = MEMBER_TABLES.get(Path(info.name).stem)
			if table is None:
				continue

			handle = tar.extractfile(info)
			if handle is None:
				continue

			started = datetime.now(UTC)
			read, copied, written = load_member(
				conn,
				handle,
				table,
				member=info.name,
				in_top=in_top,
				in_random=in_random,
			)
			results.append(
				MemberResult(
					dump_file=dump_path.name,
					member=info.name,
					target_table=table,
					in_top=in_top,
					in_random=in_random,
					rows_read=read,
					rows_copied=copied,
					rows_written=written,
					started_at=started,
					finished_at=datetime.now(UTC),
				)
			)

	return results


def record(conn: psycopg.Connection, results: list[MemberResult]) -> None:
	"""Append ingest_log rows."""
	columns = [f.name for f in fields(MemberResult)]

	with conn.cursor() as cur:
		cur.executemany(
			sql.SQL("INSERT INTO ingest_log ({columns}) VALUES ({placeholders})").format(
				columns=_identifiers(columns),
				placeholders=sql.SQL(", ").join(sql.Placeholder(c) for c in columns),
			),
			[{c: getattr(r, c) for c in columns} for r in results],
		)

	conn.commit()
