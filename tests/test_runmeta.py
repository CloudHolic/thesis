"""Provenance records written beside artifacts."""

import json

from thesis import runmeta

RAW = {
	"db": {"dsn": "postgresql://someone:secret@dbhost:5432/thesis"},
	"data": {"response": "acc", "keys": [4, 7], "link_threshold": 10},
}


def test_build_records_the_config_and_the_environment():
	meta = runmeta.build(script="02_linking_report.py", config_raw=RAW)

	assert meta["script"] == "02_linking_report.py"
	assert meta["config"]["data"]["keys"] == [4, 7]
	assert meta["config"]["data"]["link_threshold"] == 10
	assert meta["python"]
	assert meta["platform"]
	assert meta["written_at"]


def test_build_records_tracked_package_versions():
	packages = runmeta.build(script="s.py", config_raw=RAW)["packages"]

	assert set(packages) == set(runmeta.TRACKED_PACKAGES)
	assert packages["polars"] != "not installed"


def test_build_strips_credentials_from_the_dsn():
	meta = runmeta.build(script="s.py", config_raw=RAW)
	dumped = json.dumps(meta)

	assert meta["config"]["db"]["dsn"] == "postgresql://dbhost/thesis"
	assert "secret" not in dumped
	assert "someone" not in dumped


def test_build_does_not_mutate_the_caller_config():
	runmeta.build(script="s.py", config_raw=RAW)

	assert RAW["db"]["dsn"] == "postgresql://someone:secret@dbhost:5432/thesis"


def test_build_tolerates_a_config_without_a_db_section():
	meta = runmeta.build(script="s.py", config_raw={"data": {"response": "acc"}})

	assert meta["config"] == {"data": {"response": "acc"}}


def test_extra_fields_are_merged():
	meta = runmeta.build(script="s.py", config_raw=RAW, extra={"effective": {"response": "score"}})

	assert meta["effective"]["response"] == "score"


def test_write_places_the_file_beside_the_artifact(tmp_path):
	artifact = tmp_path / "table.parquet"
	artifact.write_bytes(b"")

	out = runmeta.write(artifact, runmeta.build(script="s.py", config_raw=RAW))

	assert out == tmp_path / "table.parquet.meta.json"
	assert json.loads(out.read_text(encoding="utf-8"))["script"] == "s.py"


def test_write_keeps_the_artifact_name_intact(tmp_path):
	artifact = tmp_path / "kcore_sweep_acc_random.parquet"
	artifact.write_bytes(b"")

	out = runmeta.write(artifact, runmeta.build(script="s.py", config_raw=RAW))

	assert out.name == "kcore_sweep_acc_random.parquet.meta.json"
	assert artifact.exists()
