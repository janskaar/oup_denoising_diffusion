from simulator import ParticleSimulator, SimulationParameters
import numpy as np
from scipy import stats
from scipy.linalg import expm
import scipy.linalg
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

num_procs = 10000

z0 = np.zeros((num_procs, 2))

zs = []

## Simulations

params = SimulationParameters(
    num_procs=num_procs,
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)

simulator = ParticleSimulator(z0, params)
simulator.simulate(2000)

zs = simulator.z
zs = zs[:30000:10][-1024:].transpose(1,0,2)

ddpm_params = get_ddpm_params(config.ddpm)
ddpm_params = {k: np.array(v) for k, v in ddpm_params.items()}

##

def get_noised_process(z, t, ddpm_params):
    noise = np.random.randn(*zs.shape)
    return q_sample(z, t, noise, ddpm_params)

z_t = get_noised_process(zs, 0, ddpm_params)

## 

# def compute_stationary_covariance(params):
#     A = jnp.array([[1 / params.tau_y, 0               ],
#                   [-1 / params.C,     1 / params.tau_x]])
#     s1 = params.sigma2_noise / (2 * A[0,0]) 
#     s2 = -( A[1,0] * params.sigma2_noise ) / ( 2 * A[0,0] * (A[0,0] + A[1,1]) )
#     s3 = ( A[1,0]**2 * params.sigma2_noise ) / ( 2 * A[0,0] * A[1,1] * (A[0,0] + A[1,1]) )
#     return  jnp.array([[s1, s2],
#                        [s2, s3]])


# def compute_ou_covariance(delta_s, params):
#     # TODO: Can add diffusion time step here, since the full temporal covariance
#     #       is linear in the stationary covariance, we can apply the diffusion
#     #       step to the stationary covariance instead of the full time covariance.
# 
#     cov_s = compute_stationary_covariance(params)
# 
#     A = jnp.array([[1 / params.tau_y, 0               ],
#                   [-1 / params.C,     1 / params.tau_x]])
# 
#     @partial(jax.vmap, in_axes=(None, 0))
#     def vectorized_expm(A, s):
#         return jexpm(- A * s)
# 
#     expAdelta = vectorized_expm(A, delta_s)
# 
#     # matmul along last two dimensions, broadcast along first (time) dimension
#     cov_row = einops.einsum(cov_s, expAdelta.transpose(0,2,1), "j k,i k l->i j l")
# 
#     # make a new, symmetric vector that we can slice for the rows in the covariance matrix
#     cov_row_sym = jnp.concatenate((jnp.flip(cov_row[1:].transpose(0,2,1), axis=0), cov_row))
# 
# 
#     @partial(jax.vmap, in_axes=(1, None))
#     @partial(jax.vmap, in_axes=(1, None))
#     @partial(jax.vmap, in_axes=(None, 0))
#     def create_block_row(row, i):
#         return jax.lax.dynamic_slice_in_dim(row, i, 1024)
# 
#     indices = jnp.arange(1024)[::-1]
# 
#     blocks = create_block_row(cov_row_sym, indices)
# 
#     cov = einops.rearrange(blocks, "c1 c2 s1 s2 -> (c1 s1) (c2 s2)")
#     return cov


# def compute_ou_cov_diffusion_t(alpha_bar_t, delta_s, params):
#     cov_ou = compute_ou_covariance(delta_s, params)
#     cov = cov_ou * alpha_bar_t + jnp.eye(2048) * (1 - alpha_bar_t)
#     return cov


delta_s = jnp.arange(1024)
cov = compute_ou_temporal_covariance(delta_s, params)

sample = stats.multivariate_normal.rvs(mean=np.zeros(2048), cov=cov, size=10)

cov_100 = compute_ou_temporal_covariance_t(ddpm_params["alphas_bar"][100], delta_s, params)

zs_100 = get_noised_process(zs, 100, ddpm_params)

##

cov_sim1 = compute_sample_temporal_covariance(zs[:,0], zs)
cov_sim_100 = compute_sample_temporal_covariance(zs_100[:,0], zs_100)

## Plot temporal correlations of the O-U process without diffusion

fig, ax = plt.subplots(2, 2, sharex=True)
fig.set_size_inches(10,10)

ax[0,0].plot(cov_sim[:,0,0], label="simulation")
ax[0,0].plot(cov[0,:1024], label="theory")
ax[0,0].legend()

ax[0,1].plot(cov_sim[:,0,1])
ax[0,1].plot(cov[0,1024:])

ax[1,0].plot(cov_sim[:,1,0])
ax[1,0].plot(cov[1024,:1024])

ax[1,1].plot(cov_sim[:,1,1])
ax[1,1].plot(cov[1024,1024:])

ax[0,0].set_xlim(0, 50)

plt.show()

## Plot temporal correlations of the O-U process at diffusion step 100

fig, ax = plt.subplots(2, 2, sharex=True)
fig.set_size_inches(10,10)

ax[0,0].plot(cov_sim_100[:,0,0], label="simulation")
ax[0,0].plot(cov_100[0,:1024], label="theory")
ax[0,0].legend()

ax[0,1].plot(cov_sim_100[:,0,1])
ax[0,1].plot(cov_100[0,1024:])

ax[1,0].plot(cov_sim_100[:,1,0])
ax[1,0].plot(cov_100[1024,:1024])

ax[1,1].plot(cov_sim_100[:,1,1])
ax[1,1].plot(cov_100[1024,1024:])

ax[0,0].set_xlim(0, 50)

plt.show()

## 
# 
# @partial(jax.vmap, in_axes=(0, 0, None, None, None))
# def forward_posterior_sample(key, x_t, alpha_bar_t, beta_t, sigma_OU):
#     n = 2048
#     sigma_q_t = 1 - alpha_bar_t
#     t1_inverse = jnp.linalg.solve(alpha_bar_t * sigma_OU + sigma_q_t * jnp.eye(n), jnp.eye(n))
# 
#     Sigma_inv = t1_inverse + (1 - beta_t) / beta_t * jnp.eye(n)
# 
#     chol_Sigma_inv = jnp.linalg.cholesky(Sigma_inv, upper=False)
#     chol_Sigma = jsp.linalg.solve_triangular(chol_Sigma_inv, jnp.eye(n), lower=True)
#     Sigma = chol_Sigma.dot(chol_Sigma.T)
#     mu = np.sqrt(1 - beta_t) / beta_t * Sigma.dot(x_t)
# 
#     sample = chol_Sigma.dot(jax.random.normal(key, shape=x_t.shape)) + mu
#     return sample

rng = jax.random.PRNGKey(123)
rng, key = jax.random.split(rng)
x_T = jax.random.normal(key, shape=(10000, 2048))
samples = [x_T]
for i in reversed(range(1000)):
    tic = time.time()

    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, len(x_T))

    cov_t = compute_ou_temporal_covariance_t(ddpm_params["alphas_bar"][i], delta_s, params)
    sample = sample_forward_posterior(keys, samples[-1], ddpm_params["alphas_bar"][i], ddpm_params["betas"][i], cov_t)
    if ( i % 50 == 0 ):
        samples.append(sample)
    toc = time.time()
    print(f"step {i}, {toc - tic:.1f} s")
    break

## 

with h5py.File("reverse_process_analytical_samples.h5", "r") as f:
    samples = f["samples"][()]

cov_reverse_0 = compute_temporal_covariances(zs_100[:,0,:], zs_100)


