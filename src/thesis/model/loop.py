"""The optax loop shared by every objective."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


@dataclass(frozen=True, slots=True)
class Fit:
	"""The best point found, and the trace that got there. `params` is a pytree."""

	params: Any
	losses: np.ndarray
	seconds: np.ndarray
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
	loss_fn: Callable[[Any, Array], Array],
	params: Any,
	*,
	optimizer: optax.GradientTransformation,
	steps: int,
	tolerance: float,
	patience: int,
	key: Array,
) -> Fit:
	"""Minimizes `loss_fn` from `params`, returning the best point found."""
	if patience < 1:
		raise ValueError(f"patience must be at least 1, got {patience}")

	params = jax.tree.map(jnp.asarray, params)
	state = optimizer.init(params)
	value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

	@jax.jit
	def update(params: Any, state: optax.OptState, grad: Any) -> tuple[Any, optax.OptState]:
		updates, state = optimizer.update(grad, state, params)
		return optax.apply_updates(params, updates), state

	losses: list[float] = []
	seconds: list[float] = []
	best_loss = np.inf
	best = params
	stalled = 0
	converged = False

	for step in range(steps):
		mark = time.monotonic()
		key, subkey = jax.random.split(key)
		loss, grad = value_and_grad(params, subkey)
		value = float(loss)
		if not np.isfinite(value):
			raise FloatingPointError(f"loss became {value} at step {step}")
		losses.append(value)

		if best_loss - value > tolerance:
			best_loss, best, stalled = value, params, 0
		else:
			stalled += 1
			if stalled >= patience:
				converged = True
				seconds.append(time.monotonic() - mark)
				break

		params, state = update(params, state, grad)
		seconds.append(time.monotonic() - mark)

	return Fit(
		params=best,
		losses=np.asarray(losses),
		seconds=np.asarray(seconds),
		best_loss=float(best_loss),
		steps=len(losses),
		converged=converged,
	)
