"""Every likelihood family the fits can select, by name."""

from __future__ import annotations

from thesis.model.family import zoi
from thesis.model.family.interior import beta
from thesis.model.family.protocol import Family

FAMILIES: dict[str, Family] = {"zoi_beta": zoi.wrap(beta)}
