"""Full-covariance variational inference over the item coordinate."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from thesis.model import likelihood
from thesis.model.family.protocol import Family
from thesis.model.fit import Objective
from thesis.utils.quadrature import Quadrature

INITIAL_LOG_SD = -2.0

INTERVAL_DRAWS = 512
INTERVAL_PROBS = (5.0, 95.0)


class Variational(NamedTuple):
	"""q(z_i) = Normal(mu_i, L_i, L_i^T), one per item. The diagonal of L is stored logged."""

	mu: Array
	cholesky: Array


def build(family: Family) -> Objective:
	"""A full-covariance variational objective over `family`'s item coordinates."""
	z_dim = family.Z_DIM
	scale = family.prior_scale
	rows, cols = np.tril_indices(z_dim)
	diag = np.flatnonzero(rows == cols)
	n_tril = rows.size

	def positive(cholesky: Array) -> Array:
		"""The packed triangle with its diagonal exponentiated."""
		return cholesky.at[:, diag].set(jnp.exp(cholesky[:, diag]))

	def matrix(cholesky: Array) -> Array:
		"""(n_items, Z_DIM, Z_DIM) lower triangular with a positive diagonal."""
		n_items = cholesky.shape[0]

		return jnp.zeros((n_items, z_dim, z_dim)).at[:, rows, cols].set(positive(cholesky))

	def init(response: np.ndarray, n_items: int) -> Variational:
		"""The same starting mean as the point estimate."""
		cholesky = np.zeros((n_items, n_tril))
		cholesky[:, diag] = INITIAL_LOG_SD

		return Variational(
			mu=jnp.asarray(family.initial_z(response, n_items)), cholesky=jnp.asarray(cholesky)
		)

	def sample(params: Variational, key: Array) -> Array:
		"""One reparameterized draw per item, shared by every person term in the step."""
		eps = jax.random.normal(key, params.mu.shape)
		return params.mu + jnp.einsum("nij,nj->ni", matrix(params.cholesky), eps)

	def kl(params: Variational) -> Array:
		"""Closed-form KL(q || Normal(0, prior_scale^2 I)), summed over items."""
		values = positive(params.cholesky)
		frobenius = jnp.sum(jnp.square(values), axis=1)
		quadratic = jnp.sum(jnp.square(params.mu), axis=1)
		log_det = 2.0 * jnp.sum(params.cholesky[:, diag], axis=1)

		return 0.5 * jnp.sum(
			(frobenius + quadratic) / scale**2 - z_dim + 2.0 * z_dim * jnp.log(scale) - log_det
		)

	def loss(
		params: Any,
		quad: Quadrature,
		responses: likelihood.Responses,
		key: Array | None = None,
	) -> Array:
		"""Negative ELBO."""
		if key is None:
			raise ValueError("the ELBO is stochastic and needs a key")

		tau = family.to_tau(sample(params, key))
		return -likelihood.log_marginal(family, tau, quad, responses).sum() + kl(params)

	def summary(params: Any, key: Array | None = None) -> dict[str, np.ndarray]:
		"""The posterior mean, a tau-space interval, and the width of each coordinate."""
		if key is None:
			raise ValueError("the ELBO is stochastic and needs a key")

		draws = jax.vmap(lambda k: sample(params, k))(jax.random.split(key, INTERVAL_DRAWS))
		tau = jnp.stack(jax.tree.leaves(family.to_tau(draws)), axis=-1)
		lower, upper = np.percentile(np.asarray(tau), INTERVAL_PROBS, axis=0)

		return {
			"z": np.asarray(params.mu),
			"tau_lower": lower,
			"tau_upper": upper,
			"log_sd": np.asarray(params.cholesky[:, diag]),
		}

	return Objective(init=init, loss=loss, summary=summary)
