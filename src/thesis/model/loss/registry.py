"""Every loss the fits can select, by name."""

from __future__ import annotations

from collections.abc import Callable

from thesis.model.family.protocol import Family
from thesis.model.fit import Objective
from thesis.model.loss import elbo
from thesis.model.loss import map as map_loss

LOSSES: dict[str, Callable[[Family], Objective]] = {"map": map_loss.build, "elbo": elbo.build}
