"""Provenance written beside every artifact."""

from __future__ import annotations

import copy
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TRACKED_PACKAGES = ("polars", "psycopg", "connectorx", "pyarrow")


def _versions() -> dict[str, str]:
	out: dict[str, str] = {}
	for name in TRACKED_PACKAGES:
		try:
			out[name] = version(name)
		except PackageNotFoundError:
			out[name] = "not installed"

	return out


def _redact(raw: dict[str, Any]) -> dict[str, Any]:
	"""Config with the DSN reduced to host and database name."""
	out = copy.deepcopy(raw)
	dsn = out.get("db", {}).get("dsn")

	if isinstance(dsn, str):
		parts = urlsplit(dsn)
		out["db"]["dsn"] = f"{parts.scheme}://{parts.hostname or ''}{parts.path}"

	return out


def build(
	*, script: str, config_raw: dict[str, Any], extra: dict[str, Any] | None = None
) -> dict[str, Any]:
	"""A provenance record for one script run."""
	return {
		"script": script,
		"written_at": datetime.now(UTC).isoformat(),
		"python": sys.version.split()[0],
		"platform": platform.platform(),
		"packages": _versions(),
		"config": _redact(config_raw),
		**(extra or {}),
	}


def write(artifact: Path, meta: dict[str, Any]) -> Path:
	"""Write `meta` next to `artifact` as <artifact>.meta.json."""
	out = artifact.with_suffix(artifact.suffix + ".meta.json")
	out.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")

	return out
