"""Parsing of osu! MySQL dump INSERT statements into typed rows."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

_TABLE_RE = re.compile(r"^INSERT INTO `([^`]+)`")

# MySQL's documented backslash escapes.
_ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}

MANIA_PLAYMODE = 3

SCORE_COLUMNS_MIN = 16
BEATMAP_COLUMNS_MIN = 28


def table_of(line: str) -> str | None:
	"""Table name of an INSERT line, or None if the line is not one."""
	m = _TABLE_RE.match(line)
	return m.group(1) if m else None


def _coerce(token: str) -> Any:
	if token == "NULL":
		return None

	try:
		return int(token)
	except ValueError:
		pass

	try:
		return float(token)
	except ValueError:
		pass


def parse_values_line(line: str) -> list[list[Any]]:
	"""Every tuple in one `INSERT ... VALUES (...),(...);` line."""
	idx = line.upper().find("VALUES")
	if not line.startswith("INSERT INTO") or idx == -1:
		return []

	rest = line[idx + len("VALUES") :]
	rows: list[list[Any]] = []
	current: list[Any] = []
	buf: list[str] = []

	in_row = False
	in_string = False
	escaped = False
	quoted = False

	def take() -> Any:
		nonlocal quoted
		raw = "".join(buf)
		value = raw if quoted else _coerce(raw.strip())
		quoted = False
		buf.clear()

		return value

	for ch in rest:
		if in_string:
			if escaped:
				buf.append(_ESCAPES.get(ch, ch))
				escaped = False
			elif ch == "\\":
				escaped = True
			elif ch == "'":
				in_string = False
			else:
				buf.append(ch)

			continue

		if ch == "'" and in_row:
			in_string = True
			quoted = True
		elif ch == "(" and not in_row:
			in_row = True
			current = []
			buf.clear()
			quoted = False
		elif ch == "," and in_row:
			current.append(take())
		elif ch == ")" and in_row:
			current.append(take())
			rows.append(current)
			in_row = False
		elif in_row:
			buf.append(ch)

	return rows


def mania_accuracy(
	*,
	count_max: int,
	count_300: int,
	count_200: int,
	count_100: int,
	count_50: int,
	count_miss: int,
) -> float:
	"""osu!mania ScoreV1 accuracy from judgment counts, in [0, 1]."""
	total = count_max + count_300 + count_200 + count_100 + count_50 + count_miss
	if total == 0:
		return 0.0

	hits = 300 * count_max + 300 * count_300 + 200 * count_200 + 100 * count_100 + 50 * count_50
	return hits / (300 * total)


def _int(value: Any, default: int = 0) -> int:
	try:
		return int(value) if value is not None else default
	except (TypeError, ValueError):
		return default


def _float(value: Any) -> float | None:
	try:
		return float(value) if value is not None else None
	except (TypeError, ValueError):
		return None


def _timestamp(value: Any) -> datetime | None:
	try:
		return datetime.fromisoformat(str(value))
	except (TypeError, ValueError):
		return None


@dataclass(frozen=True, slots=True)
class ScoreRow:
	score_id: int
	user_id: int
	beatmap_id: int
	rank: str
	count_max: int
	count_300: int
	count_200: int
	count_100: int
	count_50: int
	count_miss: int
	score: int
	accuracy: float
	mods: int
	pp: float | None
	date: datetime
	in_top: bool
	in_random: bool

	def as_tuple(self) -> tuple:
		return tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=True, slots=True)
class BeatmapRow:
	beatmap_id: int
	beatmapset_id: int | None
	keys: int
	star_rating: float | None
	length: int
	play_count: int
	pass_count: int
	bpm: float | None
	version: str

	def as_tuple(self) -> tuple:
		return tuple(getattr(self, f.name) for f in fields(self))


def to_score_row(row: list[Any], *, in_top: bool, in_random: bool) -> ScoreRow | None:
	"""One `osu_scores_mania_high` dump row, or None when unusable."""
	if len(row) < SCORE_COLUMNS_MIN:
		return None

	score_id = _int(row[0])
	date = _timestamp(row[14])
	if score_id == 0 or date is None:
		return None

	count_50, count_100, count_300 = _int(row[6]), _int(row[7]), _int(row[8])
	count_miss, count_max, count_200 = _int(row[9]), _int(row[10]), _int(row[11])

	return ScoreRow(
		score_id=score_id,
		user_id=_int(row[2]),
		beatmap_id=_int(row[1]),
		rank=str(row[5]) if row[5] is not None else "",
		count_max=count_max,
		count_300=count_300,
		count_200=count_200,
		count_100=count_100,
		count_50=count_50,
		count_miss=count_miss,
		score=_int(row[3]),
		accuracy=mania_accuracy(
			count_max=count_max,
			count_300=count_300,
			count_200=count_200,
			count_100=count_100,
			count_50=count_50,
			count_miss=count_miss,
		),
		mods=_int(row[13]),
		pp=_float(row[15]),
		date=date,
		in_top=in_top,
		in_random=in_random,
	)


def to_beatmap_row(row: list[Any]) -> BeatmapRow | None:
	"""One mania `osu_beatmaps` dump row, or None when unusable or not mania."""
	if len(row) < BEATMAP_COLUMNS_MIN or _int(row[16]) != MANIA_PLAYMODE:
		return None

	beatmap_id = _int(row[0])
	if beatmap_id == 0:
		return None

	return BeatmapRow(
		beatmap_id=beatmap_id,
		beatmapset_id=_int(row[1]) if row[1] is not None else None,
		keys=round(_float(row[13]) or 0.0),
		star_rating=_float(row[19]),
		length=_int(row[6]),
		play_count=_int(row[21]),
		pass_count=_int(row[22]),
		bpm=_float(row[27]),
		version=str(row[5]) if row[5] is not None else "",
	)
