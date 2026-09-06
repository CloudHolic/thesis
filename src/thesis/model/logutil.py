"""Log-space primitives that JAX does not provide and that the naive forms get wrong."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

_LOG2 = 0.6931471805599453


def log1mexp(u: Array) -> Array:
    """log(1 - exp(u)) for u < 0."""
    u = jnp.minimum(u, -jnp.finfo(jnp.asarray(u).dtype).tiny)
    near = u > -_LOG2

    return jnp.where(
        near,
        jnp.log(-jnp.expm1(jnp.where(near, u, -1.0))),
        jnp.log1p(-jnp.exp(jnp.where(near, -1.0, u)))
    )


def log_diff_exp(high: Array, low: Array) -> Array:
    """log(exp(high) - exp(low)), requiring high > low."""
    return high + log1mexp(low - high)