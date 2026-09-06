"""Schema and view application, against a throwaway schema.

Every test runs inside a scratch schema that is dropped afterwards, so the
production tables are never touched.
"""

import uuid

import psycopg
import pytest
from psycopg import sql

from thesis import config
from thesis.db import apply

pytestmark = pytest.mark.db

# Mod bits, so the fixtures read as the mods a player would have picked.
NF, EZ, HD, HR, DT, HT, NC = 1, 2, 8, 16, 64, 256, 512

VIEW_COLUMNS = {
	"user_id",
	"beatmap_id",
	"rate_group",
	"response",
	"keys",
	"in_top",
	"in_random",
}


@pytest.fixture
def dsn():
	return config.load().db.dsn


@pytest.fixture
def scratch_schema(dsn):
	name = f"test_apply_{uuid.uuid4().hex[:8]}"
	with psycopg.connect(dsn) as conn, conn.cursor() as cur:
		cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
		conn.commit()

	yield name

	with psycopg.connect(dsn) as conn, conn.cursor() as cur:
		cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
		conn.commit()


@pytest.fixture
def conn(dsn, scratch_schema):
	with psycopg.connect(dsn) as c:
		apply.apply_all(c, schema=scratch_schema)
		with c.cursor() as cur:
			cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(scratch_schema)))
		yield c


def objects_in(conn, schema: str) -> set[str]:
	with conn.cursor() as cur:
		cur.execute(
			"SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
			(schema,),
		)
		return {r[0] for r in cur.fetchall()}


def columns_of(conn, schema: str, table: str) -> set[str]:
	with conn.cursor() as cur:
		cur.execute(
			"SELECT column_name FROM information_schema.columns "
			"WHERE table_schema = %s AND table_name = %s",
			(schema, table),
		)
		return {r[0] for r in cur.fetchall()}


def seed(conn, beatmaps, scores):
	"""Insert (beatmap_id, keys) rows and score rows.

	A score row is (score_id, user_id, beatmap_id, mods, score, accuracy).
	"""
	with conn.cursor() as cur:
		cur.executemany("INSERT INTO beatmaps (beatmap_id, keys) VALUES (%s, %s)", beatmaps)
		cur.executemany(
			"INSERT INTO scores "
			"(score_id, user_id, beatmap_id, mods, score, accuracy, rank, date, in_random) "
			"VALUES (%s, %s, %s, %s, %s, %s, 'A', '2026-03-04 05:06:07', TRUE)",
			scores,
		)
	conn.commit()


def rows(conn, view: str, **where):
	clause = " AND ".join(f"{k} = %s" for k in where)
	sql_text = f"SELECT user_id, rate_group, response, keys FROM {view}"
	if clause:
		sql_text += f" WHERE {clause}"
	with conn.cursor() as cur:
		cur.execute(sql_text, tuple(where.values()))
		return cur.fetchall()


def test_apply_creates_the_tables_and_views(conn, scratch_schema):
	found = objects_in(conn, scratch_schema)

	assert {"scores", "beatmaps", "ingest_log"} <= found
	assert {"v_response_acc", "v_response_score"} <= found


def test_apply_is_idempotent(conn, scratch_schema):
	apply.apply_all(conn, schema=scratch_schema)
	apply.apply_all(conn, schema=scratch_schema)

	assert {"scores", "beatmaps", "ingest_log"} <= objects_in(conn, scratch_schema)


def test_views_expose_the_contracted_columns(conn, scratch_schema):
	for view in ("v_response_acc", "v_response_score"):
		assert columns_of(conn, scratch_schema, view) == VIEW_COLUMNS


def test_key_count_comes_from_the_beatmap(conn):
	seed(
		conn,
		[(1001, 4), (1002, 7)],
		[(1, 10, 1001, 0, 800000, 0.95), (2, 10, 1002, 0, 700000, 0.93)],
	)

	got = {r[0]: r[3] for r in rows(conn, "v_response_score", user_id=10, beatmap_id=1001)}
	assert got == {10: 4}

	got = {r[0]: r[3] for r in rows(conn, "v_response_score", user_id=10, beatmap_id=1002)}
	assert got == {10: 7}


def test_nofail_score_is_doubled(conn):
	seed(conn, [(1001, 4)], [(1, 10, 1001, NF, 480000, 0.99)])

	assert rows(conn, "v_response_score", user_id=10) == [(10, "NM", pytest.approx(0.96), 4)]


def test_halftime_score_is_doubled(conn):
	seed(conn, [(1001, 4)], [(1, 10, 1001, HT, 450000, 0.97)])

	assert rows(conn, "v_response_score", user_id=10) == [(10, "HT", pytest.approx(0.90), 4)]


def test_nofail_and_halftime_multipliers_compose(conn):
	seed(conn, [(1001, 4)], [(1, 10, 1001, NF | HT, 240000, 0.96)])

	assert rows(conn, "v_response_score", user_id=10) == [(10, "HT", pytest.approx(0.96), 4)]


def test_doubletime_score_is_not_scaled(conn):
	seed(conn, [(1001, 4)], [(1, 10, 1001, DT, 730000, 0.94)])

	assert rows(conn, "v_response_score", user_id=10) == [(10, "DT", pytest.approx(0.73), 4)]


def test_nightcore_lands_in_the_doubletime_group(conn):
	seed(
		conn,
		[(1001, 4)],
		[(1, 10, 1001, NC, 700000, 0.93), (2, 11, 1001, DT | NC, 710000, 0.94)],
	)

	assert rows(conn, "v_response_score", user_id=10)[0][1] == "DT"
	assert rows(conn, "v_response_score", user_id=11)[0][1] == "DT"


def test_best_play_is_chosen_by_the_normalized_score(conn):
	# The NoFail run scores less raw but more once its 0.5 multiplier is undone.
	seed(
		conn,
		[(1001, 4)],
		[(1, 10, 1001, 0, 500000, 0.91), (2, 10, 1001, NF, 480000, 0.99)],
	)

	assert rows(conn, "v_response_score", user_id=10) == [(10, "NM", pytest.approx(0.96), 4)]


def test_best_play_for_accuracy_is_chosen_by_accuracy(conn):
	seed(
		conn,
		[(1001, 4)],
		[(1, 10, 1001, 0, 900000, 0.91), (2, 10, 1001, NF, 480000, 0.99)],
	)

	assert rows(conn, "v_response_acc", user_id=10) == [(10, "NM", pytest.approx(0.99), 4)]


def test_rate_groups_are_deduplicated_separately(conn):
	seed(
		conn,
		[(1001, 4)],
		[
			(1, 10, 1001, 0, 800000, 0.95),
			(2, 10, 1001, DT, 730000, 0.94),
			(3, 10, 1001, HT, 450000, 0.97),
		],
	)

	got = {r[1]: r[2] for r in rows(conn, "v_response_score", user_id=10)}
	assert got == {
		"NM": pytest.approx(0.80),
		"DT": pytest.approx(0.73),
		"HT": pytest.approx(0.90),
	}


def test_hidden_is_kept_and_left_uncorrected(conn):
	seed(conn, [(1001, 4)], [(1, 10, 1001, HD, 900000, 0.96)])

	assert rows(conn, "v_response_score", user_id=10) == [(10, "NM", pytest.approx(0.90), 4)]


@pytest.mark.parametrize("mods", [EZ, HR, HD | HR, 1024, 536870912])
def test_disallowed_mods_are_filtered_out(conn, mods):
	seed(conn, [(1001, 4)], [(1, 10, 1001, mods, 800000, 0.95)])

	assert rows(conn, "v_response_score", user_id=10) == []
	assert rows(conn, "v_response_acc", user_id=10) == []


def test_scores_on_unknown_beatmaps_are_dropped(conn):
	# Converted charts from other rulesets never enter the beatmaps table, so the
	# join is what excludes them.
	seed(conn, [(1001, 4)], [(1, 10, 9999, 0, 800000, 0.95)])

	assert rows(conn, "v_response_score", user_id=10) == []
	assert rows(conn, "v_response_acc", user_id=10) == []
