"""
Computes the forward process posterior covariance matrices for the O-U process with
parameters defined here, over all diffusion time steps t=1,2,...,1000, and saves them to file.
"""

from simulator import SimulationParameters
import numpy as np
import os, time, h5py
from functools import partial
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from jax.scipy.linalg import expm as jexpm
import einops

from diffusion import get_ddpm_params
from default_config import config as config

np.random.seed(123)

params = SimulationParameters(
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)

ddpm_params = get_ddpm_params(config.ddpm)
ddpm_params = {k: np.array(v) for k, v in ddpm_params.items()}

##

def compute_stationary_covariance(params):
    A = jnp.array([[1 / params.tau_y, 0               ],
                  [-1 / params.C,     1 / params.tau_x]])
    s1 = params.sigma2_noise / (2 * A[0,0]) 
    s2 = -( A[1,0] * params.sigma2_noise ) / ( 2 * A[0,0] * (A[0,0] + A[1,1]) )
    s3 = ( A[1,0]**2 * params.sigma2_noise ) / ( 2 * A[0,0] * A[1,1] * (A[0,0] + A[1,1]) )
    return  jnp.array([[s1, s2],
                       [s2, s3]])


def compute_ou_covariance(delta_s, params):
    # TODO: Can add diffusion time step here, since the full temporal covariance
    #       is linear in the stationary covariance, we can apply the diffusion
    #       step to the stationary covariance instead of the full time covariance.

    cov_s = compute_stationary_covariance(params)

    A = jnp.array([[1 / params.tau_y, 0               ],
                  [-1 / params.C,     1 / params.tau_x]])

    @partial(jax.vmap, in_axes=(None, 0))
    def vectorized_expm(A, s):
        return jexpm(- A * s)

    expAdelta = vectorized_expm(A, delta_s)

    # matmul along last two dimensions, broadcast along first (time) dimension
    cov_row = einops.einsum(cov_s, expAdelta.transpose(0,2,1), "j k,i k l->i j l")

    # make a new, symmetric vector that we can slice for the rows in the covariance matrix
    cov_row_sym = jnp.concatenate((jnp.flip(cov_row[1:].transpose(0,2,1), axis=0), cov_row))


    @partial(jax.vmap, in_axes=(1, None))
    @partial(jax.vmap, in_axes=(1, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def create_block_row(row, i):
        return jax.lax.dynamic_slice_in_dim(row, i, 1024)

    indices = jnp.arange(1024)[::-1]

    blocks = create_block_row(cov_row_sym, indices)

    cov = einops.rearrange(blocks, "c1 c2 s1 s2 -> (c1 s1) (c2 s2)")
    return cov


def compute_ou_cov_diffusion_t(alpha_bar_t, delta_s, params):
    cov_ou = compute_ou_covariance(delta_s, params)
    cov = cov_ou * alpha_bar_t + jnp.eye(2048) * (1 - alpha_bar_t)
    return cov


delta_s = jnp.arange(1024)


##

@jax.jit
def forward_posterior_chol_cov(alpha_bar_t, beta_t, sigma_OU):
    n = 2048
    sigma_q_t = 1 - alpha_bar_t
    t1_inverse = jnp.linalg.solve(alpha_bar_t * sigma_OU + sigma_q_t * jnp.eye(n), jnp.eye(n))

    Sigma_inv = t1_inverse + (1 - beta_t) / beta_t * jnp.eye(n)

    chol_Sigma_inv = jnp.linalg.cholesky(Sigma_inv, upper=False)
    chol_Sigma = jsp.linalg.solve_triangular(chol_Sigma_inv, jnp.eye(n), lower=True)
    return chol_Sigma

chol_covs = []
for i in range(1000):
    tic = time.time()
    cov_t = compute_ou_cov_diffusion_t(ddpm_params["alphas_bar"][i], delta_s, params)
    chol_cov = forward_posterior_chol_cov(ddpm_params["alphas_bar"][i], ddpm_params["betas"][i], cov_t)
    chol_covs.append(chol_cov) 
    toc = time.time()
    print(f"step {i}, {toc - tic:.1f} s")
chol_covs = np.array(chol_covs)

with h5py.File("forward_process_posterior_covariances.h5", "a") as f:
    f.create_dataset("chol_cov", data=chol_covs)

    f.attrs["sigma_noise"] = 5.
    f.attrs["tau_x"] = 4.
    f.attrs["tau_y"] = 5.
    f.attrs["C"] = 50.

