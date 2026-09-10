"""Every synthetic generator, by the family name it generates for."""

from __future__ import annotations

from thesis.diagnostics.generate import zoi_beta
from thesis.diagnostics.generate.protocol import Generator

GENERATORS: dict[str, Generator] = {"zoi_beta": zoi_beta}
