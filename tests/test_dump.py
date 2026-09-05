"""osu! dump INSERT-line parsing and row conversion."""

from dataclasses import fields
from datetime import datetime

import pytest

from thesis.db import dump

# osu_scores_mania_high column order, every value distinct so that an index off
# by one shows up as a wrong field rather than a plausible number.
SCORE_DUMP_ROW = [
	910000000001,  # 0  score_id
	2001,  # 1  beatmap_id
	3001,  # 2  user_id
	987654,  # 3  score
	1234,  # 4  maxcombo
	"SH",  # 5  rank
	11,  # 6  count50
	7,  # 7  count100
	3,  # 8  count300
	13,  # 9  countmiss
	2,  # 10 countgeki -> MAX
	5,  # 11 countkatu -> 200
	0,  # 12 perfect
	64,  # 13 enabled_mods
	"2026-03-04 05:06:07",  # 14 date
	123.5,  # 15 pp
	0,  # 16 replay
	0,  # 17 hidden
	"KR",  # 18 country_acronym
]


def beatmap_dump_row(**overrides):
	row = [0] * 28
	row[0] = 2001  # beatmap_id
	row[1] = 4001  # beatmapset_id
	row[5] = "Another"  # version
	row[6] = 187  # total_length
	row[13] = 7.0  # diff_size -> key count
	row[16] = 3  # playmode
	row[17] = 1  # approved
	row[19] = 5.43  # difficultyrating
	row[21] = 9871  # playcount
	row[22] = 3210  # passcount
	row[27] = 174.5  # bpm

	for index, value in overrides.items():
		row[int(index)] = value
	return row


# Parsing ----------------------------------------------------------------------


def test_parses_multiple_tuples_from_one_line():
	line = "INSERT INTO `t` VALUES (1,'a',2.5),(3,'b',-4);"
	assert dump.parse_values_line(line) == [[1, "a", 2.5], [3, "b", -4]]


def test_parses_null_as_none():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (7,NULL,'x');") == [[7, None, "x"]]


def test_parses_escaped_quote_inside_string():
	bs, q, bt = chr(92), chr(39), chr(96)
	line = f"INSERT INTO {bt}t{bt} VALUES (1,{q}O{bs}{q}Reilly{q});"
	assert dump.parse_values_line(line) == [[1, "O'Reilly"]]


def test_parses_escaped_backslash():
	# built from chr() so that transcribing this test cannot lose an escape level
	bs, q, bt = chr(92), chr(39), chr(96)
	line = f"INSERT INTO {bt}t{bt} VALUES (1,{q}a{bs}{bs}b{q});"
	assert dump.parse_values_line(line) == [[1, f"a{bs}b"]]


def test_parses_parenthesis_inside_string():
	line = "INSERT INTO `t` VALUES (1,'Song (TV Size)',9);"
	assert dump.parse_values_line(line) == [[1, "Song (TV Size)", 9]]


def test_parses_comma_inside_string():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (1,'a,b',9);") == [[1, "a,b", 9]]


def test_parses_empty_string():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (1,'',9);") == [[1, "", 9]]


def test_parses_scientific_notation():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (1,1.5e-3);")[0][1] == pytest.approx(
		0.0015
	)


def test_quoted_value_that_looks_numeric_stays_a_string():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (1,'404');") == [[1, "404"]]


def test_quoted_value_that_looks_like_null_stays_a_string():
	assert dump.parse_values_line("INSERT INTO `t` VALUES (1,'NULL');") == [[1, "NULL"]]


def test_non_insert_line_yields_nothing():
	assert dump.parse_values_line("-- a comment") == []
	assert dump.parse_values_line("CREATE TABLE `t` (id INT);") == []


def test_table_name_is_extracted():
	assert dump.table_of("INSERT INTO `osu_beatmaps` VALUES (1);") == "osu_beatmaps"
	assert dump.table_of("CREATE TABLE `x` (id INT);") is None


# Accuracy ---------------------------------------------------------------------


def test_mania_accuracy_weights_each_judgement():
	acc = dump.mania_accuracy(
		count_max=2, count_300=3, count_200=5, count_100=7, count_50=11, count_miss=13
	)

	hits = 300 * 2 + 300 * 3 + 200 * 5 + 100 * 7 + 50 * 11
	assert acc == pytest.approx(hits / (300 * 41))


def test_mania_accuracy_weighs_max_the_same_as_300():
	# the 305 weight is ScoreV2's and 320 is the score formula's hit value
	only_max = dump.mania_accuracy(
		count_max=9, count_300=0, count_200=0, count_100=0, count_50=0, count_miss=0
	)
	only_300 = dump.mania_accuracy(
		count_max=0, count_300=9, count_200=0, count_100=0, count_50=0, count_miss=0
	)

	assert only_max == only_300 == pytest.approx(1.0)


def test_mania_accuracy_of_an_all_miss_play_is_zero():
	assert (
		dump.mania_accuracy(
			count_max=0, count_300=0, count_200=0, count_100=0, count_50=0, count_miss=7
		)
		== 0.0
	)


def test_mania_accuracy_of_an_empty_play_is_zero():
	assert (
		dump.mania_accuracy(
			count_max=0, count_300=0, count_200=0, count_100=0, count_50=0, count_miss=0
		)
		== 0.0
	)


# Score rows -------------------------------------------------------------------


def test_score_row_maps_dump_columns_to_named_fields():
	got = dump.to_score_row(SCORE_DUMP_ROW, in_top=False, in_random=True)

	assert got.score_id == 910000000001
	assert got.user_id == 3001
	assert got.beatmap_id == 2001
	assert got.rank == "SH"
	assert got.score == 987654
	assert got.mods == 64
	assert got.pp == pytest.approx(123.5)
	assert got.date == datetime(2026, 3, 4, 5, 6, 7)
	assert (got.in_top, got.in_random) == (False, True)


def test_score_row_maps_the_dump_judgement_names_to_mania_ones():
	got = dump.to_score_row(SCORE_DUMP_ROW, in_top=True, in_random=False)

	assert got.count_max == 2  # countgeki
	assert got.count_300 == 3
	assert got.count_200 == 5  # countkatu
	assert got.count_100 == 7
	assert got.count_50 == 11
	assert got.count_miss == 13


def test_score_row_computes_accuracy_from_the_judgements():
	got = dump.to_score_row(SCORE_DUMP_ROW, in_top=False, in_random=True)

	assert got.accuracy == pytest.approx(
		dump.mania_accuracy(
			count_max=2, count_300=3, count_200=5, count_100=7, count_50=11, count_miss=13
		)
	)


def test_score_row_rejects_a_short_row():
	assert dump.to_score_row([1, 2, 3], in_top=True, in_random=False) is None


def test_score_row_rejects_a_missing_date():
	row = list(SCORE_DUMP_ROW)
	row[14] = None
	assert dump.to_score_row(row, in_top=True, in_random=False) is None


def test_score_row_rejects_a_malformed_date():
	row = list(SCORE_DUMP_ROW)
	row[14] = "not-a-date"
	assert dump.to_score_row(row, in_top=True, in_random=False) is None


def test_score_row_rejects_a_zero_score_id():
	row = list(SCORE_DUMP_ROW)
	row[0] = 0
	assert dump.to_score_row(row, in_top=True, in_random=False) is None


def test_score_row_tuple_follows_the_field_order():
	got = dump.to_score_row(SCORE_DUMP_ROW, in_top=False, in_random=True)
	names = [f.name for f in fields(dump.ScoreRow)]

	assert names == [
		"score_id",
		"user_id",
		"beatmap_id",
		"rank",
		"count_max",
		"count_300",
		"count_200",
		"count_100",
		"count_50",
		"count_miss",
		"score",
		"accuracy",
		"mods",
		"pp",
		"date",
		"in_top",
		"in_random",
	]
	assert got.as_tuple() == tuple(getattr(got, n) for n in names)


# Beatmap rows -----------------------------------------------------------------


def test_beatmap_row_maps_dump_columns_to_named_fields():
	got = dump.to_beatmap_row(beatmap_dump_row())

	assert got.beatmap_id == 2001
	assert got.beatmapset_id == 4001
	assert got.version == "Another"
	assert got.length == 187
	assert got.keys == 7
	assert got.star_rating == pytest.approx(5.43)
	assert got.play_count == 9871
	assert got.pass_count == 3210
	assert got.bpm == pytest.approx(174.5)


@pytest.mark.parametrize("playmode", [0, 1, 2])
def test_beatmap_row_rejects_other_rulesets(playmode):
	assert dump.to_beatmap_row(beatmap_dump_row(**{"16": playmode})) is None


def test_beatmap_row_rejects_a_short_row():
	assert dump.to_beatmap_row([0] * 10) is None


def test_beatmap_row_tolerates_columns_appended_by_newer_dumps():
	# The 2026-09 dump appends lazer_only at index 28; every index we read is
	# unchanged, so the row must still convert.
	row = beatmap_dump_row() + [0]

	got = dump.to_beatmap_row(row)

	assert got is not None
	assert (got.beatmap_id, got.keys, got.bpm) == (2001, 7, pytest.approx(174.5))


def test_beatmap_row_rejects_a_zero_beatmap_id():
	assert dump.to_beatmap_row(beatmap_dump_row(**{"0": 0})) is None


@pytest.mark.parametrize(("diff_size", "expected"), [(4.0, 4), (7.0, 7), (6.7, 7), (10.2, 10)])
def test_beatmap_row_rounds_the_key_count(diff_size, expected):
	assert dump.to_beatmap_row(beatmap_dump_row(**{"13": diff_size})).keys == expected


def test_beatmap_row_tuple_follows_the_field_order():
	got = dump.to_beatmap_row(beatmap_dump_row())
	names = [f.name for f in fields(dump.BeatmapRow)]

	assert names == [
		"beatmap_id",
		"beatmapset_id",
		"keys",
		"star_rating",
		"length",
		"play_count",
		"pass_count",
		"bpm",
		"version",
	]
	assert got.as_tuple() == tuple(getattr(got, n) for n in names)
