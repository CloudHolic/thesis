"""Molenaar's ZOI Beta-IRT."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

SITES: tuple[str, ...] = ("log_a", "b", "omega", "gamma_0", "gamma_1")


class Posterior(NamedTuple):
    """Item parameter summaries, and the diagnostics that say whether to trust them."""
    mean: dict[str, np.ndarray]
    median: dict[str, np.ndarray]
    lower: dict[str, np.ndarray]
    upper: dict[str, np.ndarray]
    r_hat: dict[str, np.ndarray]
    divergences: int


def mode(item: jnp.ndarray, person: jnp.ndarray, y: jnp.ndarray, n_items: int, n_persons: int) -> None:
    """The model: theta is sampled, gamma_1 is truncated below gamma_0."""
    with numpyro.plate("persons", n_persons):
        theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))
    with numpryo.plate("items", n_items):
        log_a = numpyro.sample("log_a", dist.Normal(0.0, 10.0))
        b = numpyro.sample("b", dist.Normal(0.0, 10.0))
        omega = numpyro.sample("omega", dist.Normal(0.0, 10.0))
        gamma_0 = numpyro.sample("gamma_0", dist.Normal(0.0, 10.0))
        gamma_1 = numpyro.sample("gamma_1", dist.TruncatedNormal(0.0, 10.0, low=gamma_0))

    a_theta = jnp.exp(log_a)[item] * theta[person]
    eta = a_theta + b[item]
    k0 = jax.nn.sigmoid(gamma_0[item] - a_theta)
    k1 = jax.nn.sigmoid(gamma_1[item] - a_theta)

    interior = (y > 0.0) & (y < 1.0)
    half_omega = omega[item] / 2.0
    shapes = dist.Beta(jnp.exp(eta / 2.0 + half_omega), jnp.exp(-eta / 2.0 + half_omega))

    # log(k1 - k0) is left in probability space.
    log_k = jnp.where(
        interior,
        jnp.log(k1 - k0) + shapes.log_prob(jnp.where(interior, y, 0.5)),
        jnp.where(y == 0.0, jnp.log(k0), jnp.log1p(-k1))
    )
    numpyro.factor("responses", log_k.sum())


def run(item: np.ndarray, person: np.ndarray, y: np.ndarray, n_items: int, n_persons: int, *, seed: int, chains: int, samples: int, warmup: int) -> Posterior:
    """NUTS over the joint (theta, item) posterior."""
    mcmc = MCMC(
        NUTS(model),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(item),
        jnp.asarray(person),
        jnp.asarray(y),
        n_items,
        n_persons,
        extra_fields=("diverging",)
    )

    stats = summary(mcmc.get_samples(group_by_chain=True))
    return Posterior(
        mean={s: np.asarray(stats[s]["mean"]) for s in SITES},
        median={s: np.asarray(stats[s]["median"]) for s in SITES},
        lower={s: np.asarray(stats[s]["5.0%"]) for s in SITES},
        upper={s: np.asarray(stats[s]["95.0%"]) for s in SITES},
        r_hat={s: np.asarray(stats[s]["r_hat"]) for s in SITES},
        divergences=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    )
