"""Marginalizing person ability out of the person-major likelihood."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.ops import segment_sum
from jax.scipy.special import logsumexp

from . import kernel
from .quadrature import Quadrature
from .transform import Tau


class Responses(NamedTuple):
	"""Responses partitioned by mixture branch, so the Beta term never sees y=0 or y=1."""

	n_persons: int  # a branch can be empty and then cannot report the count
	interior_item: Array
	interior_person: Array
	interior_log_y: Array
	interior_log1m_y: Array
	zero_item: Array
	zero_person: Array
	one_item: Array
	one_person: Array


def split_by_branch(
	item_index: np.ndarray, person_index: np.ndarray, response: np.ndarray, n_persons: int
) -> Responses:
	"""Partitions responses into the interior, zero and one branches, preserving order."""
	if not ((response >= 0.0) & (response <= 1.0)).all():
		raise ValueError("response must be in [0, 1]")

	at_zero = response == 0.0
	at_one = response == 1.0
	interior = ~(at_zero | at_one)
	y = response[interior]

	return Responses(
		n_persons=n_persons,
		interior_item=jnp.asarray(item_index[interior]),
		interior_person=jnp.asarray(person_index[interior]),
		interior_log_y=jnp.asarray(np.log(y)),
		interior_log1m_y=jnp.asarray(np.log1p(-y)),
		zero_item=jnp.asarray(item_index[at_zero]),
		zero_person=jnp.asarray(person_index[at_zero]),
		one_item=jnp.asarray(item_index[at_one]),
		one_person=jnp.asarray(person_index[at_one]),
	)


def log_marginal(tau: Tau, quad: Quadrature, responses: Responses) -> Array:
	"""logsumexp_q [log w_q + sum_{i in O_p} log k(y_pi | theta_q, tau_i)], per person."""
	grid = kernel.build_grid(tau, quad.nodes)
	n_persons = responses.n_persons

	totals = segment_sum(
		kernel.log_k_interior(
			grid,
			responses.interior_item,
			responses.interior_log_y,
			responses.interior_log1m_y,
		),
		responses.interior_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		kernel.log_k_zero(grid, responses.zero_item),
		responses.zero_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		kernel.log_k_one(grid, responses.one_item),
		responses.one_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)

	return logsumexp(quad.log_weights[None, :] + totals, axis=1)
