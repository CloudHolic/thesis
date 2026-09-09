"""Full-covariance variational inference over the item coordinate."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model import encoder, likelihood

PRIOR_SCALE = 10.0

INITIAL_LOG_SD = -2.0

INTERVAL_DRAWS = 512
INTERVAL_PROBS = (5.0, 95.0)

_ROWS, _COLS = np.tril_indices(likelihood.Z_DIM)
_DIAG = np.flatnonzero(_ROWS == _COLS)
_N_TRIL = _ROWS.size


class Variational(NamedTuple):
	"""q(z_i) = Normal(mu_i, L_i, L_i^T), one per item. The diagonal of L is stored logged."""

	mu: Array
	cholesky: Array


def _positive(cholesky: Array) -> Array:
	"""The packed triangle with its diagonal exponentiated."""
	return cholesky.at[:, _DIAG].set(jnp.exp(cholesky[:, _DIAG]))


def _matrix(cholesky: Array) -> Array:
	"""(n_items, Z_DIM, Z_DIM) lower triangular with a positive diagonal."""
	n_items = cholesky.shape[0]
	return (
		jnp.zeros((n_items, likelihood.Z_DIM, likelihood.Z_DIM))
		.at[:, _ROWS, _COLS]
		.set(_positive(cholesky))
	)


def init(response: np.ndarray, n_items: int) -> Variational:
	"""Same starting mean as the point estimate."""
	cholesky = np.zeros((n_items, _N_TRIL))
	cholesky[:, _DIAG] = INITIAL_LOG_SD

	return Variational(
		mu=jnp.asarray(encoder.initial_z(response, n_items)), cholesky=jnp.asarray(cholesky)
	)


def sample(params: Variational, key: Array) -> Array:
	"""One reparameterized draw per item, shared by every person term in the step."""
	eps = jax.random.normal(key, params.mu.shape)
	return params.mu + jnp.einsum("nij,nj->ni", _matrix(params.cholesky), eps)


def kl(params: Variational) -> Array:
	"""Closed-form KL(q || Normal(0, PRIOR_SCALE^2 I)), summed over items."""
	values = _positive(params.cholesky)
	frobenius = jnp.sum(jnp.square(values), axis=1)
	quadratic = jnp.sum(jnp.square(params.mu), axis=1)
	log_det = 2.0 * jnp.sum(params.cholesky[:, _DIAG], axis=1)

	return 0.5 * jnp.sum(
		(frobenius + quadratic) / PRIOR_SCALE**2
		- likelihood.Z_DIM
		+ 2.0 * likelihood.Z_DIM * jnp.log(PRIOR_SCALE)
		- log_det
	)


def loss(
	params: Variational,
	quad: likelihood.Quadrature,
	responses: likelihood.Responses,
	key: Array | None = None,
) -> Array:
	"""Negative ELBO."""
	if key is None:
		raise ValueError("zoi_elbo is stochastic and needs a key")

	tau = likelihood.to_tau(sample(params, key))
	return -likelihood.log_marginal(tau, quad, responses).sum() + kl(params)


def summary(params: Variational, key: Array | None = None) -> dict[str, np.ndarray]:
	"""The posterior mean, a tau-space interval, and the width of each coordinate."""
	if key is None:
		raise ValueError("zoi_elbo is stochastic and needs a key")

	draws = jax.vmap(lambda k: sample(params, k))(jax.random.split(key, INTERVAL_DRAWS))
	tau = jnp.stack(likelihood.to_tau(draws), axis=-1)
	lower, upper = np.percentile(np.asarray(tau), INTERVAL_PROBS, axis=0)

	return {
		"z": np.asarray(params.mu),
		"tau_lower": lower,
		"tau_upper": upper,
		"log_sd": np.asarray(params.cholesky[:, _DIAG]),
	}
