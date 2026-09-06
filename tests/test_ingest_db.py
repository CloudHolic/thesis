"""Loading a dump member into PostgreSQL, against a throwaway schema.

load_member walks the archive member line by line, so a list of byte strings
stands in for it and the whole COPY-then-upsert path runs without a tar file or
a real dump.
"""

import uuid

import psycopg
import pytest
from psycopg import sql

from thesis import config
from thesis.db import apply, ingest

pytestmark = pytest.mark.db

SCORE_HEADER = [
	"CREATE TABLE `osu_scores_mania_high` (",
	"  `score_id` bigint(20) unsigned NOT NULL,",
	"  `beatmap_id` mediumint(8) unsigned NOT NULL,",
	"  `user_id` mediumint(8) unsigned NOT NULL,",
	"  `score` int(11) NOT NULL,",
	"  `maxcombo` smallint(5) unsigned NOT NULL,",
	"  `rank` varchar(2) NOT NULL,",
	"  `count50` smallint(5) unsigned NOT NULL,",
	"  `count100` smallint(5) unsigned NOT NULL,",
	"  `count300` smallint(5) unsigned NOT NULL,",
	"  `countmiss` smallint(5) unsigned NOT NULL,",
	"  `countgeki` smallint(5) unsigned NOT NULL,",
	"  `countkatu` smallint(5) unsigned NOT NULL,",
	"  `perfect` tinyint(1) NOT NULL,",
	"  `enabled_mods` int(11) NOT NULL,",
	"  `date` timestamp NOT NULL,",
	"  `pp` float DEFAULT NULL,",
	"  `replay` tinyint(1) NOT NULL,",
	"  `hidden` tinyint(1) NOT NULL,",
	"  `country_acronym` varchar(2) NOT NULL,",
	"  PRIMARY KEY (`score_id`)",
	") ENGINE=InnoDB;",
]


def score_values(score_id, beatmap_id=2001, user_id=3001, score=987654, mods=0):
	"""One osu_scores_mania_high tuple in dump column order."""
	return (
		f"({score_id},{beatmap_id},{user_id},{score},1234,'SH',"
		f"11,7,3,13,2,5,0,{mods},'2026-03-04 05:06:07',123.5,0,0,'KR')"
	)


def member(lines):
	"""Stand-in for an archive member.

	An iterator, not a list: load_member consumes the header and then keeps
	reading the same handle, which only works because file iteration is
	stateful. A list would replay every line from the start.
	"""
	return iter([f"{line}\n".encode() for line in lines])


def score_member(*tuples):
	insert = "INSERT INTO `osu_scores_mania_high` VALUES " + ",".join(tuples) + ";"
	return member([*SCORE_HEADER, insert])


@pytest.fixture
def dsn():
	return config.load().db.dsn


@pytest.fixture
def conn(dsn):
	name = f"test_ingest_{uuid.uuid4().hex[:8]}"
	with psycopg.connect(dsn) as setup, setup.cursor() as cur:
		cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
		setup.commit()

	with psycopg.connect(dsn) as c:
		apply.apply_all(c, schema=name)
		with c.cursor() as cur:
			cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(name)))
		yield c

	with psycopg.connect(dsn) as teardown, teardown.cursor() as cur:
		cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name)))
		teardown.commit()


def stored(conn, columns="score_id, user_id, count_max, count_200, mods, in_top, in_random"):
	with conn.cursor() as cur:
		cur.execute(f"SELECT {columns} FROM scores ORDER BY score_id")
		return cur.fetchall()


def test_load_member_copies_and_upserts(conn):
	read, copied, written = ingest.load_member(
		conn,
		score_member(score_values(1), score_values(2, user_id=3002)),
		ingest.SCORE_TABLE,
		member="osu_scores_mania_high.sql",
		in_top=False,
		in_random=True,
	)

	assert (read, copied, written) == (2, 2, 2)
	assert stored(conn) == [
		(1, 3001, 2, 5, 0, False, True),
		(2, 3002, 2, 5, 0, False, True),
	]


def test_judgement_columns_land_where_the_parser_says(conn):
	ingest.load_member(
		conn,
		score_member(score_values(1)),
		ingest.SCORE_TABLE,
		member="m.sql",
		in_top=True,
		in_random=False,
	)

	with conn.cursor() as cur:
		cur.execute(
			"SELECT count_max, count_300, count_200, count_100, count_50, count_miss "
			"FROM scores WHERE score_id = 1"
		)
		# countgeki, count300, countkatu, count100, count50, countmiss from the tuple
		assert cur.fetchone() == (2, 3, 5, 7, 11, 13)


def test_accuracy_is_computed_on_the_way_in(conn):
	ingest.load_member(
		conn,
		score_member(score_values(1)),
		ingest.SCORE_TABLE,
		member="m.sql",
		in_top=True,
		in_random=False,
	)

	with conn.cursor() as cur:
		cur.execute("SELECT accuracy FROM scores WHERE score_id = 1")
		hits = 300 * 2 + 300 * 3 + 200 * 5 + 100 * 7 + 50 * 11
		assert cur.fetchone()[0] == pytest.approx(hits / (300 * 41))


def test_pool_flags_merge_across_dumps(conn):
	# the same score in the random dump and then the top dump
	for in_top, in_random in ((False, True), (True, False)):
		ingest.load_member(
			conn,
			score_member(score_values(1)),
			ingest.SCORE_TABLE,
			member="m.sql",
			in_top=in_top,
			in_random=in_random,
		)

	assert stored(conn) == [(1, 3001, 2, 5, 0, True, True)]


def test_reloading_the_same_member_is_idempotent(conn):
	for _ in range(2):
		read, copied, written = ingest.load_member(
			conn,
			score_member(score_values(1), score_values(2, user_id=3002)),
			ingest.SCORE_TABLE,
			member="m.sql",
			in_top=False,
			in_random=True,
		)

	assert (read, copied, written) == (2, 2, 2)
	assert len(stored(conn)) == 2


def test_duplicate_score_ids_within_one_member_are_folded(conn):
	read, copied, written = ingest.load_member(
		conn,
		score_member(score_values(1), score_values(1)),
		ingest.SCORE_TABLE,
		member="m.sql",
		in_top=False,
		in_random=True,
	)

	# both tuples are copied into staging, DISTINCT ON keeps one
	assert (read, copied) == (2, 2)
	assert written == 1
	assert len(stored(conn)) == 1


def test_rows_the_converter_rejects_are_counted_but_not_written(conn):
	broken = score_values(3).replace("'2026-03-04 05:06:07'", "'not-a-date'")

	read, copied, written = ingest.load_member(
		conn,
		score_member(score_values(1), broken),
		ingest.SCORE_TABLE,
		member="m.sql",
		in_top=False,
		in_random=True,
	)

	assert (read, copied, written) == (2, 1, 1)
	assert len(stored(conn)) == 1


def test_a_reordered_dump_is_refused_before_anything_is_written(conn):
	shifted = list(SCORE_HEADER)
	shifted.insert(shifted.index("  `score` int(11) NOT NULL,"), "  `surprise` int(11) NOT NULL,")
	insert = "INSERT INTO `osu_scores_mania_high` VALUES " + score_values(1) + ";"

	with pytest.raises(ingest.DumpSchemaError, match="column order changed"):
		ingest.load_member(
			conn,
			member([*shifted, insert]),
			ingest.SCORE_TABLE,
			member="osu_scores_mania_high.sql",
			in_top=False,
			in_random=True,
		)

	assert stored(conn) == []


def test_record_writes_one_ingest_log_row_per_member(conn):
	results = [
		ingest.MemberResult(
			dump_file="d.tar.bz2",
			member="osu_scores_mania_high.sql",
			target_table=ingest.SCORE_TABLE,
			in_top=False,
			in_random=True,
			rows_read=7,
			rows_copied=5,
			rows_written=4,
			started_at=__import__("datetime").datetime(2026, 3, 4, 5, 6, 7),
			finished_at=__import__("datetime").datetime(2026, 3, 4, 5, 8, 9),
		)
	]

	ingest.record(conn, results)

	with conn.cursor() as cur:
		cur.execute("SELECT dump_file, rows_read, rows_copied, rows_written FROM ingest_log")
		assert cur.fetchall() == [("d.tar.bz2", 7, 5, 4)]
