"""The optax loop shared by every objective."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


@dataclass(frozen=True, slots=True)
class Fit:
	z: np.ndarray
	losses: np.ndarray
	best_loss: float
	steps: int
	converged: bool


def build(name: str, learning_rate: float) -> optax.GradientTransformation:
	"""Resolves any optax optimizer by name: "adam", "adamw", "sgd", "rmsprop", ..."""
	if name.startswith("_") or not hasattr(optax, name):
		raise ValueError(f"optax has no optimizer named {name!r}")

	factory = getattr(optax, name)
	if not callable(factory):
		raise ValueError(f"optax.{name} is not callable")

	try:
		optimizer = factory(learning_rate)
	except Exception as error:
		raise ValueError(f"optax.{name} does not build from a learning rate alone") from error

	if not isinstance(optimizer, optax.GradientTransformation):
		raise ValueError(f"optax.{name} returned {type(optimizer).__name__}, not an optimizer")

	return optimizer


def run(
	loss_fn: Callable[[Array], Array],
	z0: np.ndarray,
	*,
	optimizer: optax.GradientTransformation,
	steps: int,
	tolerance: float,
	patience: int,
) -> Fit:
	"""Minimizes `loss_fn` from `z0`, returning the best point found."""
	if patience < 1:
		raise ValueError(f"patience must be at least 1, got {patience}")

	z = jnp.asarray(z0)
	state = optimizer.init(z)
	value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

	@jax.jit
	def update(z: Array, state: optax.OptState, grad: Array) -> tuple[Array, optax.OptState]:
		updates, state = optimizer.update(grad, state, z)
		return cast(Array, optax.apply_updates(z, updates)), state

	losses: list[float] = []
	best_loss = np.inf
	best_z = z
	stalled = 0
	converged = False

	for step in range(steps):
		loss, grad = value_and_grad(z)
		value = float(loss)
		if not np.isfinite(value):
			raise FloatingPointError(f"loss became {value} at step {step}")
		losses.append(value)

		if best_loss - value > tolerance:
			best_loss, best_z, stalled = value, z, 0
		else:
			stalled += 1
			if stalled >= patience:
				converged = True
				break

		z, state = update(z, state, grad)

	return Fit(
		z=np.asarray(best_z),
		losses=np.asarray(losses),
		best_loss=float(best_loss),
		steps=len(losses),
		converged=converged,
	)
