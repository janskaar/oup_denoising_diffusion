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

def compute_stationary_covariance(params):
    """
    Computes the stationary covariance of the O-U process with parameters.
    The naming from the notes are as follows:
        a_1 = 1 / tau_y
        a_2 = -1 / c
        a_3 = 1 / tau_x
        b_0 = sqrt(sigma2_noise)
    """
    A = jnp.array([[1 / params.tau_y, 0               ],
                  [-1 / params.c,     1 / params.tau_x]])
    s1 = params.sigma2_noise / (2 * A[0,0]) 
    s2 = -( A[1,0] * params.sigma2_noise ) / ( 2 * A[0,0] * (A[0,0] + A[1,1]) )
    s3 = ( A[1,0]**2 * params.sigma2_noise ) / ( 2 * A[0,0] * A[1,1] * (A[0,0] + A[1,1]) )
    return  jnp.array([[s1, s2],
                       [s2, s3]])


def compute_ou_temporal_covariance(delta_s, params):
    """
    Computes the full covariance of the O-U process with time lags delta_s.
    """
    # TODO: Can add diffusion time step here, since the full temporal covariance
    #       is linear in the stationary covariance, we can apply the diffusion
    #       step to the stationary covariance instead of the full time covariance.

    cov_s = compute_stationary_covariance(params)

    A = jnp.array([[1 / params.tau_y, 0               ],
                  [-1 / params.c,     1 / params.tau_x]])

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


def compute_ou_temporal_covariance_t(alpha_bar_t, delta_s, params):
    """
    Computes the full covariance of the O-U process with time lags delta_s, after
    t diffusion steps
    """
    cov_ou = compute_ou_temporal_covariance(delta_s, params)
    cov = cov_ou * alpha_bar_t + jnp.eye(2048) * (1 - alpha_bar_t)
    return cov


##

@partial(jax.vmap, in_axes=(None, 1))
def compute_sample_temporal_covariance(x0, xt):
    """
    Computes the temporal covariance of a sample. It is assumed
    that it's a stationary process, and the covariances is computed
    as Cov(x(0), x(t)), i.e. the between the first time step and 
    all time steps.

    xt is assumed to have shape (num_samples, time_steps)
    x0 is assumed to have shape (num_samples)
    """
    return (x0[:,:,None] * xt[:,None,:]).mean(0)


@partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def sample_forward_posterior(key, x_t, t, sigma_OU, ddpm_params):
    """
    Sample the posterior forward process q(x_(t-1) | x_t, theta), where theta
    are the parameters of the O-U process x_0. sigma_OU is the covariance of x_0.
    """
    n = 2048
    sigma_q_tm1 = 1 - ddpm_params["alphas_bar"][t-1]
    alpha_bar_tm1 = ddpm_params["alphas_bar"][t-1]
    beta_t = ddpm_params["betas"][t]

    sigma_q_tm1 = 1 - alpha_bar_tm1
    t1_inverse = jnp.linalg.solve(alpha_bar_tm1 * sigma_OU + sigma_q_tm1 * jnp.eye(n), jnp.eye(n))

    Sigma_inv = t1_inverse + (1 - beta_t) / beta_t * jnp.eye(n)

    chol_Sigma_inv = jnp.linalg.cholesky(Sigma_inv, upper=False)
    chol_Sigma = jsp.linalg.solve_triangular(chol_Sigma_inv, jnp.eye(n), lower=True)
    Sigma = chol_Sigma.dot(chol_Sigma.T)
    mu = np.sqrt(1 - beta_t) / beta_t * Sigma.dot(x_t)

    sample = chol_Sigma.dot(jax.random.normal(key, shape=x_t.shape)) + mu
    return sample

@partial(jax.vmap, in_axes=(0, None, None, None))
def compute_params_forward_posterior(x_t, t, sigma_OU, ddpm_params):#alpha_bar_t, beta_t, sigma_OU):
    """
    Compute the parameters mu and Sigma of the the posterior forward process distribution 
    q(x_(t-1) | x_t, theta), where theta are the parameters of the O-U process x_0.
    sigma_OU is the covariance of x_0.
    """
    n = 2048
    sigma_q_tm1 = 1 - ddpm_params["alphas_bar"][t-1]
    alpha_bar_tm1 = ddpm_params["alphas_bar"][t-1]
    beta_t = ddpm_params["betas"][t]
    t1_inverse = jnp.linalg.solve(alpha_bar_tm1 * sigma_OU + sigma_q_tm1 * jnp.eye(n), jnp.eye(n))

    Sigma_inv = t1_inverse + (1 - beta_t) / beta_t * jnp.eye(n)

    chol_Sigma_inv = jnp.linalg.cholesky(Sigma_inv, upper=False)
    chol_Sigma = jsp.linalg.solve_triangular(chol_Sigma_inv, jnp.eye(n), lower=True)
    Sigma = chol_Sigma.dot(chol_Sigma.T)
    mu = jnp.sqrt(1 - beta_t) / beta_t * Sigma.dot(x_t)

    return mu, Sigma

@partial(jax.vmap, in_axes=(0, 0, None, None))
def compute_params_forward_conditional_posterior(x_0, x_t, t, ddpm_params):
    alpha_bar_tm1 = ddpm_params["alphas_bar"][t-1]
    alpha_bar_t = ddpm_params["alphas_bar"][t]
    beta_t = ddpm_params["betas"][t]

    at = 1 - alpha_bar_t
    atm1 = 1 - alpha_bar_tm1
    mu_tilde = ( jnp.sqrt(alpha_bar_tm1) * beta_t ) / at * x_0 + ( jnp.sqrt(alpha_bar_t) * (1 - alpha_bar_tm1) ) / at * x_t
    Sigma_tilde = ( atm1 / at ) * beta_t * np.eye(2048)
    return mu_tilde, Sigma_tilde

