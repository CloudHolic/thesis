"""Configuration loading."""

import subprocess
import sys

import pytest

from thesis import config

TEMPLATE = """
[db]
dsn = "postgresql://someone:secret@dbhost:5432/thesis"

[paths]
artifacts = "{artifacts}"

[data]
response = "score"
pool = "all"
keys = [4, 7, 10]
link_threshold = 7

[data.diagnostics]
link_thresholds = [3, 9]
kcore_min_items = [11, 22]
kcore_min_users = [33, 44]
"""


def write_config(tmp_path, text=None, artifacts=None):
	path = tmp_path / "config.toml"
	body = text if text is not None else TEMPLATE
	path.write_text(
		body.format(artifacts=(artifacts or (tmp_path / "arts")).as_posix())
		if "{artifacts}" in body
		else body,
		encoding="utf-8",
	)
	return path


@pytest.fixture
def config_file(tmp_path):
	return write_config(tmp_path)


def test_loads_every_section(config_file):
	cfg = config.load(config_file)

	assert cfg.db.dsn == "postgresql://someone:secret@dbhost:5432/thesis"
	assert cfg.data.response == "score"
	assert cfg.data.pool == "all"
	assert cfg.data.keys == (4, 7, 10)
	assert cfg.data.link_threshold == 7
	assert cfg.data.diagnostics.link_thresholds == (3, 9)
	assert cfg.data.diagnostics.kcore_min_items == (11, 22)
	assert cfg.data.diagnostics.kcore_min_users == (33, 44)


def test_explicit_path_beats_the_environment(config_file, monkeypatch, tmp_path):
	monkeypatch.setenv("THESIS_CONFIG", str(tmp_path / "does-not-exist.toml"))
	assert config.load(config_file).path == config_file


def test_environment_variable_is_used_when_no_path_given(config_file, monkeypatch):
	monkeypatch.setenv("THESIS_CONFIG", str(config_file))
	assert config.load().path == config_file


def test_missing_file_fails_loudly(tmp_path, monkeypatch):
	monkeypatch.setenv("THESIS_CONFIG", str(tmp_path / "absent.toml"))
	with pytest.raises(config.ConfigError, match="config.example.toml"):
		config.load()


def test_missing_section_is_named(tmp_path, monkeypatch):
	path = write_config(tmp_path, '[db]\ndsn = "postgresql://h/d"\n')
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match=r"missing section \[data\]"):
		config.load()


def test_missing_key_inside_a_present_section_is_named(tmp_path, monkeypatch):
	path = write_config(tmp_path, TEMPLATE.replace('response = "score"\n', ""))
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match=r"missing \[data\] response"):
		config.load()


def test_missing_nested_section_is_named(tmp_path, monkeypatch):
	path = write_config(tmp_path, TEMPLATE[: TEMPLATE.index("[data.diagnostics]")])
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match=r"data\.diagnostics"):
		config.load()


def test_unknown_response_is_rejected(tmp_path, monkeypatch):
	path = write_config(tmp_path, TEMPLATE.replace('response = "score"', 'response = "pp"'))
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match="response"):
		config.load()


def test_unknown_pool_is_rejected(tmp_path, monkeypatch):
	path = write_config(tmp_path, TEMPLATE.replace('pool = "all"', 'pool = "everything"'))
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match="pool"):
		config.load()


def test_non_integer_in_a_list_is_rejected(tmp_path, monkeypatch):
	path = write_config(tmp_path, TEMPLATE.replace("keys = [4, 7, 10]", 'keys = [4, "seven"]'))
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match="keys"):
		config.load()


def test_empty_list_is_rejected(tmp_path, monkeypatch):
	path = write_config(
		tmp_path, TEMPLATE.replace("kcore_min_items = [11, 22]", "kcore_min_items = []")
	)
	monkeypatch.setenv("THESIS_CONFIG", str(path))
	with pytest.raises(config.ConfigError, match="kcore_min_items"):
		config.load()


def test_relative_artifacts_path_resolves_against_the_repo_root(tmp_path, monkeypatch):
	path = tmp_path / "config.toml"
	path.write_text(TEMPLATE.replace("{artifacts}", "artifacts"), encoding="utf-8")
	monkeypatch.setenv("THESIS_CONFIG", str(path))

	assert config.load().paths.artifacts == config.REPO_ROOT / "artifacts"


def test_absolute_artifacts_path_is_kept(config_file, tmp_path):
	assert config.load(config_file).paths.artifacts == tmp_path / "arts"


def test_ensure_dirs_creates_the_tree(config_file):
	cfg = config.load(config_file)
	cfg.ensure_dirs()

	assert cfg.paths.artifacts.is_dir()
	assert cfg.osu_cache.is_dir()


def test_raw_keeps_the_parsed_document_for_provenance(config_file):
	assert config.load(config_file).raw["data"]["keys"] == [4, 7, 10]


def test_shipped_example_is_loadable(tmp_path, monkeypatch):
	example = config.REPO_ROOT / config.EXAMPLE_FILENAME
	copied = tmp_path / "config.toml"
	copied.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
	monkeypatch.setenv("THESIS_CONFIG", str(copied))

	cfg = config.load()

	# the point is that the shipped template parses and validates, not what any
	# particular knob is set to
	assert cfg.data.response in ("acc", "score")
	assert cfg.data.pool in ("random", "top", "all")
	assert cfg.data.keys
	assert cfg.data.link_threshold > 0
	assert cfg.data.diagnostics.kcore_min_items


def test_config_does_not_import_jax():
	# a subprocess, because another test in this session may already have loaded jax
	code = "import thesis.config, sys; print('jax' in sys.modules)"
	out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

	assert out.stdout.strip() == "False"
