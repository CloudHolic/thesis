"""Marginalizing person ability out of the person-major likelihood."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.lax import stop_gradient
from jax.ops import segment_sum
from jax.scipy.special import logsumexp

from . import kernel, laplace
from .quadrature import Quadrature
from .transform import Tau

_SQRT2 = 1.4142135623730951
_LOG_SQRT_2PI = 0.9189385332046727


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


def _log_likelihood(tau: Tau, responses: Responses, theta: Array) -> Array:
	"""Sum of log k over each person's responses. `theta` is (n_persons, n_nodes)."""
	n_persons = responses.n_persons

	totals = segment_sum(
		kernel.log_k_interior(
			kernel.gather(tau, responses.interior_item),
			theta[responses.interior_person],
			responses.interior_log_y,
			responses.interior_log1m_y,
		),
		responses.interior_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		kernel.log_k_zero(kernel.gather(tau, responses.zero_item), theta[responses.zero_person]),
		responses.zero_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)
	totals += segment_sum(
		kernel.log_k_one(kernel.gather(tau, responses.one_item), theta[responses.one_person]),
		responses.one_person,
		num_segments=n_persons,
		indices_are_sorted=True,
	)

	return totals


def log_marginal(tau: Tau, quad: Quadrature, responses: Responses) -> Array:
	"""Marginal log-likelihood per person, by Gauss-Hermite centred on each posterior."""
	mode, sd = laplace.fit(
		lambda t: _log_likelihood(tau, responses, t[:, None])[:, 0] - 0.5 * jnp.square(t),
		responses.n_persons,
	)

	# The exact integral does not depend on where the nodes sit,
	# so the neglected path through the node positions is second order.
	mode = stop_gradient(mode)
	sd = stop_gradient(sd)

	theta = mode[:, None] + _SQRT2 * sd[:, None] * quad.x[None, :]
	log_phi = -0.5 * jnp.square(theta) - _LOG_SQRT_2PI

	return jnp.log(_SQRT2 * sd) + logsumexp(
		quad.log_w[None, :]
		+ jnp.square(quad.x)[None, :]
		+ log_phi
		+ _log_likelihood(tau, responses, theta),
		axis=1,
	)
