"""Running a loss against responses, and everything the run has to report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model.family.protocol import Family
from thesis.utils.quadrature import Quadrature, gauss_hermite

from . import difficulty, likelihood, loop


class Objective(Protocol):
	"""What a loss module provides."""

	# Positional-only: what a loss module calls its own parameters is its business.
	def init(self, response: np.ndarray, n_items: int, /) -> Any: ...

	def loss(
		self,
		params: Any,
		quad: Quadrature,
		responses: likelihood.Responses,
		key: Array | None = None,
		/,
	) -> Array: ...

	def summary(self, params: Any, key: Array | None = None, /) -> dict[str, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class Settings:
	quadrature_nodes: int
	optimizer: str
	learning_rate: float
	steps: int
	tolerance: float
	patience: int
	seed: int


@dataclass(frozen=True, slots=True)
class Fitted:
	"""One fit and everything derived from it."""

	reported: dict[str, np.ndarray]
	losses: np.ndarray
	seconds: np.ndarray
	best_loss: float
	steps: int
	converged: bool
	held_out_log_likelihood: float
	theta_mode: np.ndarray
	theta_usable: np.ndarray
	item_theta_low: np.ndarray
	item_theta_high: np.ndarray
	quantities: dict[str, np.ndarray]
	extrapolated: dict[str, np.ndarray]

	@property
	def step_seconds(self) -> float:
		"""The steady-state cost. The first step carries the jit compile."""
		return float(np.median(self.seconds[1:])) if self.steps > 1 else float("nan")


def run(
	objective: Objective,
	*,
	family: Family,
	item_index: np.ndarray,
	person_index: np.ndarray,
	response: np.ndarray,
	n_items: int,
	n_persons: int,
	settings: Settings,
	held_out: np.ndarray | None = None,
) -> Fitted:
	"""Fit on the training cells and report."""
	train = np.ones(response.size, dtype=bool) if held_out is None else ~held_out
	quad = gauss_hermite(settings.quadrature_nodes)
	training = likelihood.split_by_branch(
		family, item_index[train], person_index[train], response[train], n_persons
	)

	def loss_fn(params: Any, key: Array) -> Array:
		return objective.loss(params, quad, training, key)

	fit = loop.run(
		loss_fn,
		objective.init(response[train], n_items),
		optimizer=loop.build(settings.optimizer, settings.learning_rate),
		steps=settings.steps,
		tolerance=settings.tolerance * n_persons,
		patience=settings.patience,
		key=jax.random.key(settings.seed),
	)

	reported = objective.summary(fit.params, jax.random.key(settings.seed + 1))
	tau = family.to_tau(jnp.asarray(reported["z"]))
	center = likelihood.find_center(family, tau, training)
	theta = np.asarray(center.mode)

	held_ll = float("nan")
	if held_out is not None and held_out.any():
		# The conditional of the held-out cells given the training ones, each integral
		# centred on its own posterior.
		everything = likelihood.split_by_branch(
			family, item_index, person_index, response, n_persons
		)
		held_ll = float(
			likelihood.log_marginal(family, tau, quad, everything).sum()
			- likelihood.log_marginal(family, tau, quad, training).sum()
		) / int(held_out.sum())

	quantities = difficulty.quantities(family, tau)
	low, high = difficulty.responder_range(item_index[train], person_index[train], theta, n_items)

	return Fitted(
		reported=reported,
		losses=fit.losses,
		seconds=fit.seconds,
		best_loss=fit.best_loss,
		steps=fit.steps,
		converged=fit.converged,
		held_out_log_likelihood=held_ll,
		theta_mode=theta,
		theta_usable=np.asarray(center.usable),
		item_theta_low=low,
		item_theta_high=high,
		quantities=quantities,
		extrapolated=difficulty.extrapolated(quantities, low, high),
	)
