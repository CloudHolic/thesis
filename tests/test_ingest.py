"""Dump archive handling that needs no database."""

from pathlib import Path

import pytest

from thesis.db import ingest

# A mysqldump CREATE TABLE block, with the index and constraint lines that must
# not be mistaken for columns.
BEATMAP_HEADER = [
	"-- MySQL dump 10.13",
	"DROP TABLE IF EXISTS `osu_beatmaps`;",
	"CREATE TABLE `osu_beatmaps` (",
	"  `beatmap_id` mediumint(8) unsigned NOT NULL,",
	"  `beatmapset_id` mediumint(8) unsigned DEFAULT NULL,",
	"  `user_id` mediumint(8) unsigned NOT NULL,",
	"  `filename` varchar(260) DEFAULT NULL,",
	"  `checksum` varchar(32) DEFAULT NULL,",
	"  `version` varchar(80) NOT NULL,",
	"  `total_length` mediumint(8) unsigned NOT NULL,",
	"  `hit_length` mediumint(8) unsigned NOT NULL,",
	"  `countTotal` smallint(5) unsigned NOT NULL,",
	"  `countNormal` smallint(5) unsigned NOT NULL,",
	"  `countSlider` smallint(5) unsigned NOT NULL,",
	"  `countSpinner` smallint(5) unsigned NOT NULL,",
	"  `diff_drain` float NOT NULL,",
	"  `diff_size` float NOT NULL,",
	"  `diff_overall` float NOT NULL,",
	"  `diff_approach` float NOT NULL,",
	"  `playmode` tinyint(3) unsigned NOT NULL,",
	"  `approved` tinyint(4) NOT NULL,",
	"  `last_update` timestamp NOT NULL,",
	"  `difficultyrating` float NOT NULL,",
	"  `max_combo` int(11) DEFAULT NULL,",
	"  `playcount` int(10) unsigned NOT NULL,",
	"  `passcount` int(10) unsigned NOT NULL,",
	"  `youtube_preview` varchar(50) DEFAULT NULL,",
	"  `score_version` tinyint(4) NOT NULL,",
	"  `osu_file_version` tinyint(4) DEFAULT NULL,",
	"  `deleted_at` timestamp NULL DEFAULT NULL,",
	"  `bpm` float DEFAULT NULL,",
	"  PRIMARY KEY (`beatmap_id`),",
	"  KEY `beatmapset_id` (`beatmapset_id`),",
	"  UNIQUE KEY `checksum` (`checksum`)",
	") ENGINE=InnoDB DEFAULT CHARSET=utf8;",
]


def header_with(*, append=None, insert_at=None, name="extra_column"):
	lines = list(BEATMAP_HEADER)
	column = f"  `{name}` int(11) NOT NULL,"

	if append is not None:
		lines.insert(lines.index("  `bpm` float DEFAULT NULL,") + 1, column)
	if insert_at is not None:
		lines.insert(lines.index("CREATE TABLE `osu_beatmaps` (") + 1 + insert_at, column)

	return lines


# Pool flags -------------------------------------------------------------------


@pytest.mark.parametrize(
	("filename", "expected"),
	[
		("2026_01_01_performance_mania_random_10000.tar.bz2", (False, True)),
		("2026_09_01_performance_mania_top_10000.tar.bz2", (True, False)),
		("mania_TOP_and_RANDOM.tar.bz2", (True, True)),
		("2026_01_01_performance_mania.tar.bz2", (False, False)),
	],
)
def test_pool_flags_come_from_the_filename(filename, expected):
	assert ingest.pool_flags(Path("/somewhere") / filename) == expected


# Declared columns -------------------------------------------------------------


def test_declared_columns_reads_the_create_table_block():
	got = ingest.declared_columns(BEATMAP_HEADER)

	assert len(got) == 28
	assert got[0] == "beatmap_id"
	assert got[13] == "diff_size"
	assert got[16] == "playmode"
	assert got[27] == "bpm"


def test_declared_columns_skips_index_and_constraint_lines():
	got = ingest.declared_columns(BEATMAP_HEADER)

	assert "PRIMARY" not in got
	assert "beatmapset_id" in got  # the column, not the KEY line of the same name
	assert got.count("beatmapset_id") == 1
	assert got.count("checksum") == 1


def test_declared_columns_is_empty_without_a_create_table_block():
	assert ingest.declared_columns(["-- just a comment", "INSERT INTO `x` VALUES (1);"]) == []


# Column verification ----------------------------------------------------------


def test_verify_accepts_the_known_layout():
	ingest.verify_columns("osu_beatmaps.sql", ingest.BEATMAP_TABLE, BEATMAP_HEADER)


def test_verify_accepts_a_column_appended_at_the_end():
	# what the 2026-09 dump does with lazer_only
	ingest.verify_columns(
		"osu_beatmaps.sql", ingest.BEATMAP_TABLE, header_with(append=True, name="lazer_only")
	)


def test_verify_rejects_a_column_inserted_in_the_middle():
	with pytest.raises(ingest.DumpSchemaError, match="column order changed"):
		ingest.verify_columns("osu_beatmaps.sql", ingest.BEATMAP_TABLE, header_with(insert_at=3))


def test_verify_names_the_indices_that_moved():
	with pytest.raises(ingest.DumpSchemaError) as excinfo:
		ingest.verify_columns("osu_beatmaps.sql", ingest.BEATMAP_TABLE, header_with(insert_at=3))

	message = str(excinfo.value)
	assert "diff_size" in message
	assert "playmode" in message


def test_verify_rejects_a_renamed_column():
	lines = [
		line.replace("`diff_size`", "`key_count`") if "diff_size" in line else line
		for line in BEATMAP_HEADER
	]

	with pytest.raises(ingest.DumpSchemaError, match="diff_size"):
		ingest.verify_columns("osu_beatmaps.sql", ingest.BEATMAP_TABLE, lines)


def test_verify_rejects_a_member_without_a_create_table_block():
	with pytest.raises(ingest.DumpSchemaError, match="cannot verify"):
		ingest.verify_columns("osu_beatmaps.sql", ingest.BEATMAP_TABLE, ["-- nothing here"])


def test_every_index_the_parser_reads_is_covered_by_the_required_map():
	# the guard only helps if it names every index dump.py actually indexes into
	assert max(ingest.REQUIRED_COLUMNS[ingest.BEATMAP_TABLE]) == 27
	assert max(ingest.REQUIRED_COLUMNS[ingest.SCORE_TABLE]) == 15
