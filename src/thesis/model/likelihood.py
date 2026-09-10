"""What gets marginalized: responses split by branch, and the integral over theta."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.ops import segment_sum

from thesis.model.family.protocol import Family
from thesis.utils.quadrature import Center, Quadrature, expect_standard_normal, newton_center


class Responses(NamedTuple):
	"""Responses partitioned by mixture branch, so the interior term never sees y=0 or y=1."""

	n_persons: int  # a branch can be empty and then cannot report the count
	interior_item: Array
	interior_person: Array
	interior_stats: tuple[Array, ...]
	zero_item: Array
	zero_person: Array
	one_item: Array
	one_person: Array


def split_by_branch(
	family: Family,
	item_index: np.ndarray,
	person_index: np.ndarray,
	response: np.ndarray,
	n_persons: int,
) -> Responses:
	"""Partitions responses into the interior, zero and one branches, preserving order."""
	if not ((response >= 0.0) & (response <= 1.0)).all():
		raise ValueError("response must be in [0, 1]")

	at_zero = response == 0.0
	at_one = response == 1.0
	interior = ~(at_zero | at_one)

	return Responses(
		n_persons=n_persons,
		interior_item=jnp.asarray(item_index[interior]),
		interior_person=jnp.asarray(person_index[interior]),
		interior_stats=tuple(
			jnp.asarray(stat)[:, None] for stat in family.interior_stats(response[interior])
		),
		zero_item=jnp.asarray(item_index[at_zero]),
		zero_person=jnp.asarray(person_index[at_zero]),
		one_item=jnp.asarray(item_index[at_one]),
		one_person=jnp.asarray(person_index[at_one]),
	)


def gather(tau: Any, items: Array) -> Any:
	"""Per-item fields become per-response ones, with a node axis ready to broadcast."""
	return jax.tree.map(lambda field: field[items][:, None], tau)


def _log_likelihood(family: Family, tau: Any, responses: Responses, theta: Array) -> Array:
	"""Sum of log k over each person's responses. `theta` is (n_persons, n_nodes)."""
	n_persons = responses.n_persons

	totals = segment_sum(
		family.log_k_interior(
			gather(tau, responses.interior_item),
			theta[responses.interior_person],
			responses.interior_stats,
		),
		responses.interior_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		family.log_k_zero(gather(tau, responses.zero_item), theta[responses.zero_person]),
		responses.zero_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		family.log_k_one(gather(tau, responses.one_item), theta[responses.one_person]),
		responses.one_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)

	return totals


def find_center(family: Family, tau: Any, responses: Responses) -> Center:
	"""Where each person's adaptive nodes belong, and who fell back to (0, 1)."""
	return newton_center(
		lambda t: _log_likelihood(family, tau, responses, t[:, None])[:, 0] - 0.5 * jnp.square(t),
		responses.n_persons,
	)


def log_marginal(
	family: Family,
	tau: Any,
	quad: Quadrature,
	responses: Responses,
	*,
	center: Center | None = None,
) -> Array:
	"""Marginal log-likelihood per person, by Gauss-Hermite centered on each posterior."""
	found = find_center(family, tau, responses) if center is None else center

	return expect_standard_normal(
		lambda theta: _log_likelihood(family, tau, responses, theta), quad, found
	)
