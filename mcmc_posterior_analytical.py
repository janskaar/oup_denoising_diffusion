import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from simulator import SimulationParameters

jax.config.update("jax_enable_x64", True)

params = SimulationParameters(
    sigma_noise=4.0,
    tau_x=7,
    tau_y=2,
    C=9,
)


def compute_stationary_covariance(p, dt):
    # helper vars
    a1 = 1 / p.tau_y
    a2 = 1 / p.C
    a3 = 1 / p.tau_x

    expa1 = jnp.exp(a1 * dt)
    expa2 = jnp.exp(a2 * dt)
    exp2a1 = jnp.exp(2 * a1 * dt)
    exp2a2 = jnp.exp(2 * a2 * dt)
    expa1pa3 = jnp.exp((a1 + a3) * dt)
    exp2a3 = jnp.exp(2 * a3 * dt)
    diffexp = expa1 - expa2

    sigma1 = (dt * p.sigma2_noise * exp2a1) / (exp2a1 - 1)

    sigma2 = (a2 * dt * expa1 * diffexp * p.sigma2_noise) / (
        (a1 - a3) * (-1 + exp2a1) * (-1 + expa1pa3)
    )

    prefactor = (dt * p.sigma2_noise) / (exp2a3 - 1)
    sigma3 = prefactor * (
        exp2a3
        + (a2**2 * diffexp**2 * (1 - expa1pa3))
        / ((a1 - a3) ** 2 * (exp2a1 - 1) * (expa1pa3 - 1))
    )

    return sigma1, sigma2, sigma3


dt = np.linspace(0.0001, 1, 1001)

sigma1s, sigma2s, sigma3s = compute_stationary_covariance(params, dt)
