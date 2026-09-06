"""Scale-linking diagnostics over a response frame."""

import polars as pl

from thesis.data import linking


def frame(rows):
	"""(user_id, item, keys) triples as a response frame."""
	return pl.DataFrame(
		rows,
		schema={"user_id": pl.Int64, "item": pl.Utf8, "keys": pl.Int64},
		orient="row",
	)


def counts_frame():
	# u1: 3x4K + 2x7K | u2: 2x4K only | u3: 1x4K + 3x7K | u4: 4x7K only
	rows = (
		[(1, f"a{i}|NM", 4) for i in range(3)]
		+ [(1, f"b{i}|NM", 7) for i in range(2)]
		+ [(2, f"c{i}|NM", 4) for i in range(2)]
		+ [(3, "d0|NM", 4)]
		+ [(3, f"e{i}|NM", 7) for i in range(3)]
		+ [(4, f"f{i}|NM", 7) for i in range(4)]
	)
	return frame(rows)


# Per-user counts --------------------------------------------------------------


def test_counts_responses_per_user_and_keymode():
	got = linking.per_user_counts(counts_frame(), keys=(4, 7)).sort("user_id")

	assert got["n_4"].to_list() == [3, 2, 1, 0]
	assert got["n_7"].to_list() == [2, 0, 3, 4]


def test_a_keymode_with_no_responses_becomes_a_column_of_zeros():
	# the pivot-free implementation must not raise on a requested but absent key
	got = linking.per_user_counts(counts_frame(), keys=(4, 7, 9))

	assert got["n_9"].to_list() == [0, 0, 0, 0]


# Threshold summary ------------------------------------------------------------


def test_summary_counts_users_meeting_the_threshold_everywhere():
	got = linking.summary(counts_frame(), keys=(4, 7), thresholds=(1,)).row(0, named=True)

	assert got["threshold"] == 1
	assert got["n_both"] == 2  # u1 and u3
	assert got["n_key_4"] == 3  # u1, u2, u3
	assert got["n_key_7"] == 3  # u1, u3, u4


def test_summary_tightens_as_the_threshold_rises():
	got = linking.summary(counts_frame(), keys=(4, 7), thresholds=(1, 2, 3))

	assert got["threshold"].to_list() == [1, 2, 3]
	assert got["n_both"].to_list() == [2, 1, 0]


# Pairwise ---------------------------------------------------------------------


def test_pairwise_diagonal_is_the_keymode_own_users():
	got = linking.pairwise(counts_frame(), keys=(4, 7), threshold=1)
	by_key = {row["key"]: row for row in got.iter_rows(named=True)}

	assert by_key[4]["n_4"] == 3
	assert by_key[7]["n_7"] == 3


def test_pairwise_off_diagonal_is_the_co_play_count():
	got = linking.pairwise(counts_frame(), keys=(4, 7), threshold=1)
	by_key = {row["key"]: row for row in got.iter_rows(named=True)}

	assert by_key[4]["n_7"] == 2  # u1 and u3
	assert by_key[7]["n_4"] == 2


def test_pairwise_is_symmetric():
	got = linking.pairwise(counts_frame(), keys=(4, 7, 9), threshold=1)
	by_key = {row["key"]: row for row in got.iter_rows(named=True)}

	for left in (4, 7, 9):
		for right in (4, 7, 9):
			assert by_key[left][f"n_{right}"] == by_key[right][f"n_{left}"]


# Connectivity -----------------------------------------------------------------


def test_a_fully_shared_item_makes_one_component():
	got = linking.components(frame([(1, "x|NM", 4), (2, "x|NM", 4), (2, "y|NM", 7)]))

	assert got.n_components == 1
	assert got.largest_users == 2
	assert got.largest_items == 2
	assert got.largest_share_of_responses == 1.0


def test_two_disjoint_groups_make_two_components():
	# no user touches both items, so the two scales never meet
	got = linking.components(frame([(1, "x|NM", 4), (2, "y|NM", 7)]))

	assert got.n_components == 2
	assert got.largest_users == 1
	assert got.largest_items == 1
	assert got.largest_share_of_responses == 0.5


def test_a_single_bridging_user_joins_the_components():
	rows = [(1, "x|NM", 4), (2, "y|NM", 7), (3, "x|NM", 4), (3, "y|NM", 7)]

	got = linking.components(frame(rows))

	assert got.n_components == 1
	assert got.largest_users == 3
	assert got.largest_items == 2


def test_components_report_which_keymodes_fall_outside():
	# 4K and 7K share user 3; the 9K island is on its own
	rows = [
		(1, "x|NM", 4),
		(3, "x|NM", 4),
		(3, "y|NM", 7),
		(2, "y|NM", 7),
		(9, "z|NM", 9),
	]

	got = linking.components(frame(rows))

	assert got.n_components == 2
	assert got.items_by_key_in_largest == {4: 1, 7: 1}
	assert got.items_by_key_outside == {9: 1}


def test_rate_groups_of_one_beatmap_are_separate_items():
	rows = [(1, "x|NM", 4), (1, "x|DT", 4)]

	got = linking.components(frame(rows))

	assert got.n_items == 2
	assert got.n_components == 1
