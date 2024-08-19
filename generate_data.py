from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from ou_diffusion_funcs import compute_ou_temporal_covariance


@struct.dataclass
class OUParams:
    sigma2_noise : float
    tau_x : float
    tau_y : float
    c : float


def sample_ou_process(key, params):
    delta_s = jnp.arange(1024)
    cov = compute_ou_temporal_covariance(delta_s, params)
    sample = jax.random.multivariate_normal(key, mean=np.zeros(2048), cov=cov)
    return sample


@partial(jax.vmap, in_axes=(0, None, None))
def sample_prior_and_ou_process(key, prior_min, prior_max):
    rng, key = jax.random.split(key)
    paramvals = jax.random.uniform(key, minval=prior_min, maxval=prior_max, shape=(4,))

    params = OUParams(sigma2_noise = paramvals[0],
                      tau_x = paramvals[1],
                      tau_y = paramvals[2],
                      c = paramvals[3])

    rng, key = jax.random.split(rng)
    sample = sample_ou_process(key, params)
    return sample, params


