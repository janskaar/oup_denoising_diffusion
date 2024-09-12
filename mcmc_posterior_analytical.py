import h5py
from functools import partial
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from flax import struct
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm
import jax.scipy as jsp
import einops
from tensorflow_probability.substrates import jax as tfp
from mcmc_functions import MCMCState, adapt_cov, proposal, mh_step, init_state, find_last_accepted_warmups
from ou_diffusion_funcs import compute_stationary_covariance, compute_ou_temporal_covariance, OUParams
tfd = tfp.distributions

jax.config.update("jax_enable_x64", True)


## 


def compute_log_prob(p, x):
    """
    Computes the log probability of `x` under the distribution of the OU-process
    with parameters `p`.
    """

    params = OUParams(sigma2_noise = p[0],
                      tau_x = p[1],
                      tau_y = p[2],
                      c = p[3])

    cov = compute_ou_temporal_covariance(jnp.arange(1024), params)
    logprob = tfd.MultivariateNormalTriL(loc=jnp.zeros(2048), scale_tril=jnp.linalg.cholesky(cov[:2048,:2048])).log_prob(x)
    return logprob


def compute_log_prob_normed_params(p, x):
    """
    Compute the log prob where the parameters have been rescaled to
    the unit cube
    """
    p = p * jnp.array([9, 9, 9, 9])
    p = p + jnp.array([1, 1, 1, 1])

    return compute_log_prob(p, x)

## 

rng = jax.random.PRNGKey(123)
p0 = OUParams(sigma2_noise=2,
              tau_x=2,
              tau_y=2,
              c=2)
cov0 = compute_ou_temporal_covariance(jnp.arange(1024), p0)
rng, key = jax.random.split(rng)
x0 = jax.random.multivariate_normal(key, mean=jnp.zeros(2048), cov=cov0)

potential_fn = partial(compute_log_prob_normed_params, x=x0)

num_chains = 10
state = init_state(jnp.tile(jnp.array([[0.5, 0.5, 0.5, 0.5]]), (num_chains, 1)), potential_fn)

warmup_states = [state]
for i in range(100):
    print(i, end="\r")
    rng, key = jax.random.split(rng)
    key = jax.random.split(key, num_chains)
    state, x, proposal_logp = mh_step(key, warmup_states[-1], True, potential_fn)
    warmup_states.append(state)

init_state = find_last_accepted_warmups(warmup_states)
prop_logps = []
xs = []
states = [init_state]
for i in range(500):
    print(i, end="\r")
    rng, key = jax.random.split(rng)
    key = jax.random.split(key, num_chains)
    state, x, proposal_logp = mh_step(key, states[-1], False, potential_fn)
    states.append(state)
    prop_logps.append(proposal_logp)
    xs.append(x)

sample = [np.array([s.x[i] for s in states if s.accepted[i]]) for i in range(num_chains)]
outfile = os.path.join(top_dir, f"mcmc_samples_run_{run_id}.h5")
with h5py.File(outfile, "a") as f:
    grp = f.create_group(str(obs_index))
    grp.create_dataset("y", data=y)
    for i in range(num_chains):
        grp.create_dataset(f"chain_{i}", data=sample[i])



##
















