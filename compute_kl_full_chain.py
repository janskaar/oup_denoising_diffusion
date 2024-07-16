import os, time, h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from flax.training import train_state
import flax.linen as nn
import optax
import orbax.checkpoint
from functools import partial

from simulator import SimulationParameters
from diffusion import get_ddpm_params
from unet import UNET
from default_config import config
from sampling import ddpm_sample_step
from diffusion import q_sample
from ou_diffusion_funcs import (
    compute_stationary_covariance,
    compute_ou_temporal_covariance,
    compute_params_forward_posterior,
    compute_params_forward_conditional_posterior,
)


ckpt_path = os.path.join("results_simple_loss_32_16", "checkpoint_2")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
params = raw_restored["model"]["params"]


model = UNET(
    start_filters=32,
    filter_mults=config.model.filter_mults,
    out_channels=config.data.channels,
    activation=nn.silu,
    encoder_start_filters=16,
    encoder_filter_mults=config.model.encoder_filter_mults,
    encoder_latent_dim=config.model.encoder_latent_dim,
    use_encoder=config.model.use_encoder,
    attention=config.model.use_attention,
)

optimizer = optax.adam(1e-3)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)


def norm_data(X, Theta):
    # mean 0, std 1
    X = X - X.mean(axis=1, keepdims=True)
    X /= X.std(axis=1, keepdims=True)

    # limit to [-1, 1]
    Theta = Theta - Theta.min(axis=0, keepdims=True)
    Theta /= Theta.max(axis=0, keepdims=True)
    Theta *= 2
    Theta -= 1
    return X, Theta

X_path = os.path.join("data", "z_fixed_points.npy")
X = np.load(X_path)
Theta_path = os.path.join("data", "theta_fixed_points.npy")
Theta = np.load(Theta_path)
X, Theta = norm_data(X, Theta)

rng = jax.random.PRNGKey(1)
rng, key = jax.random.split(rng)
x_batch = X[:, 1024:2048]
condition = X[:, 1024:2048]
ddpm_params = get_ddpm_params(config.ddpm)


@jax.vmap
def KL_mvn(mu0, Sigma0, mu1, Sigma1):
    k = len(Sigma1)
    Sigma1_inv = jnp.linalg.solve(Sigma1, np.eye(k))
    t1 = jnp.trace(Sigma1_inv.dot(Sigma0))
    dist = mu1 - mu0
    t2 = dist.T.dot(Sigma1_inv).dot(dist)
    s0, logdet0 = jnp.linalg.slogdet(Sigma0)
    s1, logdet1 = jnp.linalg.slogdet(Sigma1)
    t4 = logdet1 - logdet0
    return 0.5 * (t1 + t2 + len(mu0) + k + t4)


def compute_unconditional_kl_full_chain(rng, state, X, condition, sigma_OU, ddpm_params):
    """
    Compute KL over all diffusion time steps.
    """

    num_samples = len(X)
    t = jnp.arange(1, 1000)

    def kl_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(carry["key"], X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t + np.zeros(num_samples, dtype=np.int32), condition)

        x_t_flat = x_t.transpose(0,2,1).reshape((-1, 2048))
        mu_q, Sigma_q = compute_params_forward_posterior(x_t_flat, t, sigma_OU, ddpm_params)

        mu_phi = (1 / jnp.sqrt(ddpm_params["alphas"][t])) * (
                x_t - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * pred
        ).transpose(0,2,1).reshape((-1, 2048))

        Sigma_phi = ddpm_params["betas"][t] * jnp.tile(jnp.eye(2048)[None], (num_samples, 1, 1))

        kl = KL_mvn(mu_q, Sigma_q, mu_phi, Sigma_phi)

        carry = {"key": rng}
        return carry, kl

    carry = {"key": rng}
    carry, ys = jax.lax.scan(kl_at_t, carry, xs=t)
    return ys


def compute_conditional_kl_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute KL over all diffusion time steps.
    """

    num_samples = len(X)
    t = jnp.arange(1, 1000)

    def kl_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(key, X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t + np.zeros(num_samples, dtype=np.int32), condition)

        x_flat = X.transpose(0,2,1).reshape((-1, 2048))
        x_t_flat = x_t.transpose(0,2,1).reshape((-1, 2048))

        mu_tilde, Sigma_tilde = compute_params_forward_conditional_posterior(x_flat, x_t_flat, t, ddpm_params)

        mu_phi = (1 / jnp.sqrt(ddpm_params["alphas"][t])) * (
                x_t - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * pred
        ).transpose(0,2,1).reshape((-1, 2048))

        Sigma_phi = ddpm_params["betas"][t] * jnp.tile(jnp.eye(2048)[None], (num_samples, 1, 1))

        kl = KL_mvn(mu_tilde, Sigma_tilde, mu_phi, Sigma_phi)

        carry = {"key": rng}
        return carry, kl

    carry = {"key": rng}
    carry, ys = jax.lax.scan(kl_at_t, carry, xs=t)
    return ys


ou_params = SimulationParameters(
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)


delta_s = jnp.arange(1024)
sigma_OU = compute_ou_temporal_covariance(delta_s, ou_params)[None]
tic = time.time()
kls_unconditional = compute_unconditional_kl_full_chain(key, state, x_batch, x_batch, sigma_OU, ddpm_params)
toc = time.time()
print("Time unconditional: ", toc - tic)
kls_conditional = compute_conditional_kl_full_chain(key, state, x_batch, x_batch, ddpm_params)
tic = time.time()
print("Time conditional: ", tic - toc)

with h5py.File("full_chain_kl.h5", "w") as f:
    f.attrs["model"] = ckpt_path
    f.attrs["X"] = X_path
    f.attrs["Theta"] = Theta_path
    f.create_dataset("kls_conditional", data=kls_conditional)
    f.create_dataset("kls_unconditional", data=kls_unconditional)

