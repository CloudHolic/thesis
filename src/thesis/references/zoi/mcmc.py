"""Molenaar's ZOI Beta-IRT, sampled with NUTS."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from thesis.references.interior import beta
from thesis.references.zoi import kernel

SITES: tuple[str, ...] = ("log_a", "b", "omega", "gamma_0", "gamma_1")


class Posterior(NamedTuple):
	"""Item parameter summaries, and the diagnostics that say whether to trust them."""

	mean: dict[str, np.ndarray]
	median: dict[str, np.ndarray]
	lower: dict[str, np.ndarray]
	upper: dict[str, np.ndarray]
	r_hat: dict[str, np.ndarray]
	divergences: int


def model(
	item: jnp.ndarray, person: jnp.ndarray, y: jnp.ndarray, n_items: int, n_persons: int
) -> None:
	"""The model: theta is sampled, gamma_1 is truncated below gamma_0."""
	with numpyro.plate("persons", n_persons):
		theta = jnp.asarray(numpyro.sample("theta", dist.Normal(0.0, 1.0)))
	with numpyro.plate("items", n_items):
		log_a = jnp.asarray(numpyro.sample("log_a", dist.Normal(0.0, 10.0)))
		b = jnp.asarray(numpyro.sample("b", dist.Normal(0.0, 10.0)))
		omega = jnp.asarray(numpyro.sample("omega", dist.Normal(0.0, 10.0)))
		gamma_0 = jnp.asarray(numpyro.sample("gamma_0", dist.Normal(0.0, 10.0)))
		gamma_1 = jnp.asarray(
			numpyro.sample("gamma_1", dist.TruncatedNormal(0.0, 10.0, low=gamma_0))
		)

	a = jnp.exp(log_a)[item]
	a_theta = a * theta[person]
	is_interior = (y > 0.0) & (y < 1.0)

	# The interior density is masked away from the atoms before it is ever formed.
	safe = jnp.where(is_interior, y, 0.5)
	tau = beta.BetaTau(a=a, b=b[item], omega=omega[item])
	log_k = jnp.where(
		is_interior,
		kernel.log_pi_b(a_theta, gamma_0[item], gamma_1[item])
		+ beta.log_f(tau, theta[person], (jnp.log(safe), jnp.log1p(-safe))),
		jnp.where(
			y == 0.0,
			kernel.log_k_zero(a_theta, gamma_0[item]),
			kernel.log_k_one(a_theta, gamma_1[item]),
		),
	)
	numpyro.factor("responses", log_k.sum())


def run(
	item: np.ndarray,
	person: np.ndarray,
	y: np.ndarray,
	n_items: int,
	n_persons: int,
	*,
	seed: int,
	chains: int,
	samples: int,
	warmup: int,
) -> Posterior:
	"""NUTS over the joint (theta, item) posterior."""
	mcmc = MCMC(
		NUTS(model),
		num_warmup=warmup,
		num_samples=samples,
		num_chains=chains,
		chain_method="sequential",
		progress_bar=False,
	)
	mcmc.run(
		jax.random.PRNGKey(seed),
		jnp.asarray(item),
		jnp.asarray(person),
		jnp.asarray(y),
		n_items,
		n_persons,
		extra_fields=("diverging",),
	)

	stats = summary(mcmc.get_samples(group_by_chain=True))
	return Posterior(
		mean={s: np.asarray(stats[s]["mean"]) for s in SITES},
		median={s: np.asarray(stats[s]["median"]) for s in SITES},
		lower={s: np.asarray(stats[s]["5.0%"]) for s in SITES},
		upper={s: np.asarray(stats[s]["95.0%"]) for s in SITES},
		r_hat={s: np.asarray(stats[s]["r_hat"]) for s in SITES},
		divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
	)
