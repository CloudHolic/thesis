"""Loading the single run-configuration file."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import domain

DEFAULT_FILENAME = "config.toml"
EXAMPLE_FILENAME = "config.example.toml"

# Assumes a source checkout or an editable installation, which is how all three environments
# install this package. THESIS_CONFIG overrides it when that fails.
REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
	"""The configuration file is missing, incomplete, or malformed."""


@dataclass(frozen=True, slots=True)
class DbConfig:
	dsn: str


@dataclass(frozen=True, slots=True)
class PathsConfig:
	artifacts: Path


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
	link_thresholds: tuple[int, ...]
	kcore_min_items: tuple[int, ...]
	kcore_min_users: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DataConfig:
	response: str
	pool: str
	keys: tuple[int, ...]
	link_threshold: int
	diagnostics: DiagnosticsConfig


@dataclass(frozen=True, slots=True)
class Config:
	path: Path
	db: DbConfig
	paths: PathsConfig
	data: DataConfig
	raw: dict[str, Any]

	@property
	def osu_cache(self) -> Path:
		return self.paths.artifacts / "osu"

	def ensure_dirs(self) -> None:
		"""Create the artifact tree if it does not exist."""
		for d in (self.paths.artifacts, self.osu_cache):
			d.mkdir(parents=True, exist_ok=True)


def resolve_path(explicit: str | os.PathLike[str] | None = None) -> Path:
	"""Which configuration file a run should read."""
	if explicit is not None:
		return Path(explicit)

	env = os.environ.get("THESIS_CONFIG")
	if env:
		return Path(env)

	return REPO_ROOT / DEFAULT_FILENAME


def load(path: str | os.PathLike[str] | None = None) -> Config:
	"""Parse and validate the configuration file."""
	resolved = resolve_path(path)
	if not resolved.is_file():
		raise ConfigError(
			f"no configuration file at {resolved}. "
			f"Copy {EXAMPLE_FILENAME} to {DEFAULT_FILENAME} and fill in the DSN, "
			f"or point THESIS_CONFIG at another file."
		)

	with resolved.open("rb") as fh:
		raw = tomllib.load(fh)

	return _build(resolved, raw)


def _require(raw: dict[str, Any], section: str, key: str) -> Any:
	node: Any = raw
	for part in section.split("."):
		node = node.get(part) if isinstance(node, dict) else None
		if node is None:
			raise ConfigError(f"missing section [{section}] in the configuration file")

	if not isinstance(node, dict):
		raise ConfigError(f"[{section}] must be a table in the configuration file")
	if key not in node:
		raise ConfigError(f"missing [{section}] {key} in the configuration file")

	return node[key]


def _ints(value: Any, section: str, key: str) -> tuple[int, ...]:
	if not isinstance(value, list) or not value:
		raise ConfigError(f"[{section}] {key} must be a non-empty list of integers")

	try:
		return tuple(int(v) for v in value)
	except (TypeError, ValueError):
		raise ConfigError(f"[{section}] {key} must contain only integers") from None


def _build(path: Path, raw: dict[str, Any]) -> Config:
	response = _require(raw, "data", "response")
	if response not in domain.VIEWS:
		raise ConfigError(f"[data] response must be one of {sorted(domain.VIEWS)}")

	pool = _require(raw, "data", "pool")
	if pool not in domain.POOLS:
		raise ConfigError(f"[data] pool must be one of {list(domain.POOLS)}")

	artifacts = Path(str(_require(raw, "paths", "artifacts")))
	if not artifacts.is_absolute():
		artifacts = REPO_ROOT / artifacts

	return Config(
		path=path,
		db=DbConfig(dsn=str(_require(raw, "db", "dsn"))),
		paths=PathsConfig(artifacts=artifacts),
		data=DataConfig(
			response=response,
			pool=pool,
			keys=_ints(_require(raw, "data", "keys"), "data", "keys"),
			link_threshold=int(_require(raw, "data", "link_threshold")),
			diagnostics=DiagnosticsConfig(
				link_thresholds=_ints(
					_require(raw, "data.diagnostics", "link_thresholds"),
					"data.diagnostics",
					"link_thresholds",
				),
				kcore_min_items=_ints(
					_require(raw, "data.diagnostics", "kcore_min_items"),
					"data.diagnostics",
					"kcore_min_items",
				),
				kcore_min_users=_ints(
					_require(raw, "data.diagnostics", "kcore_min_users"),
					"data.diagnostics",
					"kcore_min_users",
				),
			),
		),
		raw=raw,
	)
