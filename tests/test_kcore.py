"""Bidirectional iterative k-core filtering and sweeps."""

import polars as pl

from thesis.data import kcore


def frame(rows):
	"""(user_id, item, keys) triples, with beatmap_id and rate_group split out."""
	df = pl.DataFrame(
		rows,
		schema={"user_id": pl.Int64, "item": pl.Utf8, "keys": pl.Int64},
		orient="row",
	)
	return df.with_columns(
		pl.col("item").str.split("|").list.get(0).cast(pl.Int64).alias("beatmap_id"),
		pl.col("item").str.split("|").list.get(1).alias("rate_group"),
	)


def two_round_frame():
	# p1: i1 i2 i3 | p2: i1 i2 i3 | p3: i1 i4 | p4: i1
	return frame(
		[
			(1, "1|NM", 4),
			(1, "2|NM", 4),
			(1, "3|NM", 4),
			(2, "1|NM", 4),
			(2, "2|NM", 4),
			(2, "3|NM", 4),
			(3, "1|NM", 4),
			(3, "4|NM", 4),
			(4, "1|NM", 4),
		]
	)


# Filtering --------------------------------------------------------------------


def test_the_core_converges_over_more_than_one_round():
	# round 1: item 4 has one response and goes; p3 and p4 then hold one item each
	# round 2: items 1-3 have two each, p1 and p2 have three each - fixed point
	got = kcore.filter_kcore(two_round_frame(), min_item=2, min_user=2)

	assert sorted(got["item"].unique().to_list()) == ["1|NM", "2|NM", "3|NM"]
	assert sorted(got["user_id"].unique().to_list()) == [1, 2]
	assert got.height == 6


def test_a_single_pass_would_not_be_enough():
	# after one pass p3 and p4 still look fine on their pre-filter counts
	one_pass = two_round_frame().filter(
		(pl.len().over("item") >= 2) & (pl.len().over("user_id") >= 2)
	)

	assert one_pass.height > kcore.filter_kcore(two_round_frame(), min_item=2, min_user=2).height


def test_threshold_of_one_keeps_everything():
	df = two_round_frame()

	assert kcore.filter_kcore(df, min_item=1, min_user=1).height == df.height


def test_an_impossible_threshold_empties_the_frame():
	assert kcore.filter_kcore(two_round_frame(), min_item=99, min_user=99).height == 0


# Sweep ------------------------------------------------------------------------


def test_sweep_reports_one_row_per_grid_point():
	got = kcore.sweep(two_round_frame(), min_items=(1, 2), min_users=(1, 2))

	assert got.height == 4
	assert set(got.columns) >= {
		"min_item",
		"min_user",
		"n_items",
		"n_persons",
		"n_obs",
		"median_resp_per_item",
		"n_rate_group_pairs",
		"n_components",
		"largest_share",
		"n_keys",
		"items_by_key",
	}


def test_sweep_records_the_converged_counts():
	got = kcore.sweep(two_round_frame(), min_items=(2,), min_users=(2,)).row(0, named=True)

	assert (got["n_items"], got["n_persons"], got["n_obs"]) == (3, 2, 6)
	assert got["median_resp_per_item"] == 2.0


def test_sweep_survives_a_grid_point_that_empties_the_frame():
	got = kcore.sweep(two_round_frame(), min_items=(99,), min_users=(99,)).row(0, named=True)

	assert got["n_items"] == 0
	assert got["n_obs"] == 0
	assert got["n_components"] == 0


def test_sweep_counts_beatmaps_surviving_in_more_than_one_rate_group():
	rows = [
		(1, "10|NM", 4),
		(2, "10|NM", 4),
		(1, "10|DT", 4),
		(2, "10|DT", 4),
		(1, "20|NM", 4),
		(2, "20|NM", 4),
	]

	got = kcore.sweep(frame(rows), min_items=(2,), min_users=(1,)).row(0, named=True)

	# beatmap 10 survives as both NM and DT; beatmap 20 only as NM
	assert got["n_rate_group_pairs"] == 1
	assert got["n_items"] == 3


def test_sweep_reports_items_per_keymode():
	rows = [
		(1, "10|NM", 4),
		(2, "10|NM", 4),
		(1, "20|NM", 7),
		(2, "20|NM", 7),
		(1, "30|NM", 7),
		(2, "30|NM", 7),
	]

	got = kcore.sweep(frame(rows), min_items=(2,), min_users=(1,)).row(0, named=True)

	assert got["n_keys"] == 2
	assert got["items_by_key"] == str({4: 1, 7: 2})


def test_sweep_reports_connectivity_at_each_grid_point():
	# two islands: users 1-2 on item 10, users 3-4 on item 20, nobody shared
	rows = [
		(1, "10|NM", 4),
		(2, "10|NM", 4),
		(3, "20|NM", 7),
		(4, "20|NM", 7),
	]

	got = kcore.sweep(frame(rows), min_items=(2,), min_users=(1,)).row(0, named=True)

	assert got["n_components"] == 2
	assert got["largest_share"] == 0.5


def test_filtering_can_split_a_graph_that_was_connected():
	# Users 1, 2, 4 and 5 sit on their own side with three responses each. User 3
	# is the only bridge and holds two, so raising min_user to three removes just
	# them and the halves come apart.
	rows = [
		(1, "10|NM", 4),
		(1, "11|NM", 4),
		(1, "12|NM", 4),
		(2, "10|NM", 4),
		(2, "11|NM", 4),
		(2, "12|NM", 4),
		(4, "20|NM", 7),
		(4, "21|NM", 7),
		(4, "22|NM", 7),
		(5, "20|NM", 7),
		(5, "21|NM", 7),
		(5, "22|NM", 7),
		(3, "10|NM", 4),
		(3, "20|NM", 7),
	]

	loose = kcore.sweep(frame(rows), min_items=(1,), min_users=(2,)).row(0, named=True)
	tight = kcore.sweep(frame(rows), min_items=(1,), min_users=(3,)).row(0, named=True)

	assert loose["n_components"] == 1
	assert loose["n_persons"] == 5

	assert tight["n_components"] == 2
	assert tight["n_persons"] == 4
	assert tight["largest_share"] == 0.5
