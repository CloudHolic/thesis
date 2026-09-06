"""Process-wide float precision for JAX, which must be set efore any array exists."""

from __future__ import annotations

import jax

PRECISIONS: tuple[str, ...] = ("x32", "x64")


def enable(precision: str) -> None:
	"""Sets the JAX x64 flag."""
	if precision not in PRECISIONS:
		raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")

	jax.config.update("jax_enable_x64", precision == "x64")
