"""The deterministic MAP from the unconstrained coordinate Z to the item parameters tau."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

Z_DIM = 5


class Tau(NamedTuple):
    """Item parameters as the likelihood uses them."""
    a: Array
    b: Array
    omega: Array
    gamma_0: Array
    gamma_1: Array


def to_tau(z: Array) -> Tau:
    """Maps z = (log a, b, omega, gamma_0, gamma_1) to tau."""
    if z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have last dimension {Z_DIM}, got {z.shape[-1]}")

    return Tau(
        a=jnp.exp(z[..., 0]),
        b=z[..., 1],
        omega=z[..., 2],
        gamma_0=z[..., 3],
        gamma_1=z[..., 3] + jnp.exp(z[..., 4])
    )


def to_z(tau: Tau) -> Array:
    """Inverts `to_tau`."""
    return jnp.stack(
        [
            jnp.log(tau.a),
            tau.b,
            tau.omega,
            tau.gamma_0,
            jnp.log(tau.gamma_1 - tau.gamma_0)
        ],
        axis=-1
    )