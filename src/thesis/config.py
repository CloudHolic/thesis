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

# Assumes a source checkout or an editable installation, which is how all three
# environments install this package. THESIS_CONFIG overrides it when that fails.
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIRS = ("dataset", "charts", "features", "diagnostics", "fits", "difficulty")


class ConfigError(RuntimeError):
	"""The configuration file is missing, incomplete, or malformed."""


@dataclass(frozen=True, slots=True)
class DbConfig:
	dsn: str


@dataclass(frozen=True, slots=True)
class PathsConfig:
	artifacts: Path


@dataclass(frozen=True, slots=True)
class DataConfig:
	response: str
	pool: str
	keys: tuple[int, ...]
	n_items: int
	min_responses_per_item: int
	min_responses_per_user: int
	min_items_per_key: int
	holdout_fraction: float
	holdout_min_remaining: int
	sample_seed: int
	holdout_seed: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
	quadrature_nodes: int
	precision: str


@dataclass(frozen=True, slots=True)
class TrainConfig:
	optimizer: str
	learning_rate: float
	steps: int
	tolerance: float
	patience: int
	seed: int


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
	chains: int
	samples: int
	warmup: int


@dataclass(frozen=True, slots=True)
class Config:
	path: Path
	db: DbConfig
	paths: PathsConfig
	data: DataConfig
	model: ModelConfig
	train: TrainConfig
	reference: ReferenceConfig
	raw: dict[str, Any]

	def directory(self, name: str) -> Path:
		"""One of the artifact subdirectories, created on demand."""
		if name not in ARTIFACT_DIRS:
			raise ValueError(f"artifact directory must be one of {ARTIFACT_DIRS}, got {name!r}")
		path = self.paths.artifacts / name
		path.mkdir(parents=True, exist_ok=True)

		return path


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
	node = raw.get(section)
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
			n_items=int(_require(raw, "data", "n_items")),
			min_responses_per_item=int(_require(raw, "data", "min_responses_per_item")),
			min_responses_per_user=int(_require(raw, "data", "min_responses_per_user")),
			min_items_per_key=int(_require(raw, "data", "min_items_per_key")),
			holdout_fraction=float(_require(raw, "data", "holdout_fraction")),
			holdout_min_remaining=int(_require(raw, "data", "holdout_min_remaining")),
			sample_seed=int(_require(raw, "data", "sample_seed")),
			holdout_seed=int(_require(raw, "data", "holdout_seed")),
		),
		model=ModelConfig(
			quadrature_nodes=int(_require(raw, "model", "quadrature_nodes")),
			precision=str(_require(raw, "model", "precision")),
		),
		train=TrainConfig(
			optimizer=str(_require(raw, "train", "optimizer")),
			learning_rate=float(_require(raw, "train", "learning_rate")),
			steps=int(_require(raw, "train", "steps")),
			tolerance=float(_require(raw, "train", "tolerance")),
			patience=int(_require(raw, "train", "patience")),
			seed=int(_require(raw, "train", "seed")),
		),
		reference=ReferenceConfig(
			chains=int(_require(raw, "reference", "chains")),
			samples=int(_require(raw, "reference", "samples")),
			warmup=int(_require(raw, "reference", "warmup")),
		),
		raw=raw,
	)
