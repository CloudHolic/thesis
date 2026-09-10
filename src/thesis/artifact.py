"""An artifact and the two files that must travel with it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from thesis import runmeta


@dataclass(frozen=True, slots=True)
class Artifact:
	"""Arrays, the report a person reads, and what belongs in the provenance record."""

	arrays: dict[str, np.ndarray]
	title: str
	preamble: str
	body: str
	extra: dict[str, Any]


def write(path: Path, artifact: Artifact, *, script: str, config_raw: dict[str, Any]) -> None:
	"""Write the arrays, the Markdown beside them, and the provenance beside both."""
	meta = runmeta.build(script=script, config_raw=config_raw, extra=artifact.extra)
	document = f"# {artifact.title}\n\n{artifact.preamble}\n\n{artifact.body}\n"

	np.savez_compressed(path, allow_pickle=False, **artifact.arrays)
	path.with_suffix(".md").write_text(document, encoding="utf-8")
	runmeta.write(path, meta)
