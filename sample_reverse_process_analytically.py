"""
Samples the analytical reverse process with the parameters given
in the script. Saves every 50 diffusion steps.
"""

from simulator import ParticleSimulator, SimulationParameters
import numpy as np
import os, time, h5py
from functools import partial
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax.scipy.linalg import expm as jexpm
import einops

from diffusion import get_ddpm_params, q_sample
from default_config import config as config
from ou_diffusion_funcs import (compute_stationary_covariance,
                                compute_ou_temporal_covariance,
                                compute_ou_temporal_covariance_t,
                                compute_sample_temporal_covariance,
                                sample_forward_posterior)


np.random.seed(123)


## OU parameters

params = SimulationParameters(
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)

## 

ddpm_params = get_ddpm_params(config.ddpm)
ddpm_params = {k: np.array(v) for k, v in ddpm_params.items()}


##

delta_s = jnp.arange(1024)

rng = jax.random.PRNGKey(123)
rng, key = jax.random.split(rng)
# sample at step T
sample = jax.random.normal(key, shape=(10000, 2048))
samples = [sample]
cov_OU = compute_stationary_covariance(delta_s, params)
for i in reversed(range(999)):
    tic = time.time()

    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, len(sample))

    sample = sample_forward_posterior(keys, sample, ddpm_params["alphas_bar"][i], ddpm_params["betas"][i+1], cov_OU)
    if ( i % 100 == 0 ):
        samples.append(sample)
    toc = time.time()
    print(f"step {i}, {toc - tic:.1f} s")

samples = np.array(samples)

with h5py.File("reverse_process_analytical_samples.h5", "a") as f:
    f.create_dataset("samples", data=samples)

    f.attrs["sigma_noise"] = 5.
    f.attrs["tau_x"] = 4.
    f.attrs["tau_y"] = 5.
    f.attrs["C"] = 50.

## 

# with h5py.File("reverse_process_analytical_samples.h5", "r") as f:
#     samples = f["samples"][()]


## 

ts = list(range(0, 1000, 100)) + [999]
i = 0 
for i, sample in enumerate(reversed(samples)):
    sample = sample.reshape((-1, 2, 1024)).transpose(0,2,1)
    cov_sample = compute_ou_temporal_covariances(sample[:,0,:], sample)
    cov_analytical = compute_ou_temporal_covariances_t(ddpm_params["alphas_bar"][ts[i]], delta_s, params)
    fig, ax = plt.subplots(2, 2, sharex=True)
    fig.set_size_inches(10,10)
    
    ax[0,0].plot(cov_sample[:,0,0], label="simulation")
    ax[0,0].plot(cov_analytical[0,:1024], label="theory")
    ax[0,0].legend()
    
    ax[0,1].plot(cov_sample[:,0,1])
    ax[0,1].plot(cov_analytical[0,1024:])
    
    ax[1,0].plot(cov_sample[:,1,0])
    ax[1,0].plot(cov_analytical[1024,:1024])
    
    ax[1,1].plot(cov_sample[:,1,1])
    ax[1,1].plot(cov_analytical[1024,1024:])
    
    ax[0,0].set_xlim(0, 50)

    fig.savefig(os.path.join("figures_reverse_process", f"reverse_process_step_{ts[i]}.png"))
    plt.close()


