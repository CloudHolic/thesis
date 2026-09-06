"""THe ZOI Beta-IRT kernel, evaluated on the (item, node) grid and gathered per response."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.nn import log_sigmoid
from jax.scipy.special import gammaln

from . import logutil
from .transform import Tau


class Grid(NamedTuple):
    """The terms that depend on (item, node) but not on the responding person."""
    log_pi_0: Array
    log_pi_b: Array
    log_pi_1: Array
    alpha: Array
    beta: Array
    log_beta_fn: Array


def build_grid(tau: Tau, nodes: Array) -> Grid:
    """Evaluates the item-by-node terms once."""
    a_theta = tau.a[:, None] * nodes[None, :]
    eta = a_theta + tau.b[:, None]
    half_omega = tau.omega[:, None] / 2.0

    log_g0 = log_sigmoid(tau.gamma_0[:, None] - a_theta)
    log_g1 = log_sigmoid(tau.gamma_1[:, None] - a_theta)

    alpha = jnp.exp(eta / 2.0 + half_omega)
    beta = jnp.exp(-eta / 2.0 + half_omega)

    return Grid(
        log_pi_0=log_g0,
        log_pi_b=logutil.log_diff_exp(log_g1, log_g0),
        log_pi_1=log_sigmoid(a_theta - tau.gamma_1[:, None]),
        alpha=alpha,
        beta=beta,
        log_beta_fn=gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    )


def log_k_interior(grid: Grid, items: Array, log_y: Array, log1m_y: Array) -> Array:
    """log k for responses strictly inside (0, 1). Shape (n_responses, n_nodes)."""
    return (
        grid.log_pi_b[items]
        + (grid.alpha[items] - 1.0) * log_y[:, None]
        + (grid.beta[items] - 1.0) * log1m_y[:, None]
        - grid.log_beta_fn[items]
    )


def log_k_zero(grid: Grid, items: Array) -> Array:
    """log k for responses at exactly 0. Shape (n_responses, n_nodes)."""
    return grid.log_pi_0[items]


def log_k_one(grid: Grid, items: Array) -> Array:
    """log k for responses at exactly 1. Shape (n_responses, n_nodes)."""
    return grid.log_pi_1[items]
