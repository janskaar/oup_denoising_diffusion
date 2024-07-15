import os
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
from diffusion import l2_loss, q_sample, simple_loss
from ou_diffusion_funcs import (
    compute_stationary_covariance,
    compute_ou_temporal_covariance,
    compute_ou_temporal_covariance_t,
    compute_sample_temporal_covariance,
    sample_forward_posterior,
    compute_params_forward_posterior,
    compute_params_forward_conditional_posterior,
)


f = os.path.join("results_simple_loss_32_32", "checkpoint_2")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(f)
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


X = np.load(os.path.join("data", "z.npy"))
Theta = np.load(os.path.join("data", "theta.npy"))
X, Theta = norm_data(X, Theta)

rng = jax.random.PRNGKey(1)
rng, key = jax.random.split(rng)
x_batch = X[:20, 1024:2048]
condition = X[:20, 1024:2048]
ddpm_params = get_ddpm_params(config.ddpm)

# def sample_loop(rng, state, shape, condition, sample_step, timesteps):
#     rng, x_rng = jax.random.split(rng)
#     # generate the initial sample (pure noise)
#     x = jax.random.normal(x_rng, shape)
#     print("Shape: ", shape)
#     # sample step
#     print("Sampling")
#     xs = []
#     x_means = []
#     for t in reversed(jnp.arange(timesteps)):
#         print(t, end="\r")
#         rng, step_rng = jax.random.split(rng)
#         # x_mean is final prediction at t=0
#         x, x_mean = sample_step(state, step_rng, x, t, condition)
#         xs.append(x)
#         x_means.append(x_mean)
#
#     return np.array(xs), np.array(x_means)
#
#
# sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
# sample_step = jax.jit(sample_step)
#
# rng, key = jax.random.split(rng)
#

# sample_shape = (20, 1024, 2)

# x, x_mean = sample_loop(
#     key, state, sample_shape, condition, sample_step, config.ddpm.timesteps
# )

##


def loss_step(rng, state, batch, t, ddpm_params):
    # run the forward diffusion process to generate noisy image x_t at timestep t
    x = batch

    # create batched timesteps: t with shape (B,)
    B, T, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jnp.zeros(B, dtype=jnp.int32) + t

    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)

    pred = state.apply_fn({"params": state.params}, x_t, batched_t, x)
    loss = l2_loss(
        pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
    ).mean(-1)

    return loss, pred, x_t


loss_step = jax.jit(loss_step)
rng, key = jax.random.split(rng)

##


def compute_loss_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute loss over all diffusion time steps.
    """

    t = jnp.tile(jnp.arange(1000)[:, None], (1, len(X)))

    def loss_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(carry["key"], X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t, condition)

        loss = l2_loss(
            pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
        )

        carry = {"key": rng}
        return carry, jnp.array([loss.reshape(x_t.shape), pred, x_t])

    carry = {"key": rng}
    carry, ys = jax.lax.scan(loss_at_t, carry, xs=t)
    return ys


rng, key = jax.random.split(rng)
loss_pred_xt = compute_loss_full_chain(key, state, x_batch, x_batch, ddpm_params)
losses = loss_pred_xt[:, 0]
preds = loss_pred_xt[:, 1]
x_ts = loss_pred_xt[:, 2]

weights = 1 / (2 * ddpm_params["alphas"] * (1 - ddpm_params["alphas_bar"]))

weighted_losses = losses * weights[:, None, None, None]

# mean of conditional forward distribution
alphas = ddpm_params["alphas"][:, None, None, None]
betas = ddpm_params["betas"][:, None, None, None]
alpha_bars = ddpm_params["betas"][:, None, None, None]
mu = 1 / np.sqrt(alphas) * (x_ts - (betas / np.sqrt(1 - alpha_bars)) * preds)


def compute_posterior_mean(x_t, x_0, ddpm_params):
    alphas = ddpm_params["alphas"][:, None, None, None]
    betas = ddpm_params["betas"][:, None, None, None]
    alpha_bars = ddpm_params["betas"][:, None, None, None]

    k0 = (np.sqrt(alpha_bars[:-1]) * betas[1:]) / (1 - alpha_bars[1:])
    kt = np.sqrt(alphas[1:]) * (1 - alpha_bars[:-1]) / (1 - alpha_bars[1:])

    return x_0 * k0 + x_t * kt


mu_tilde = compute_posterior_mean(x_ts[1:], x_batch[None], ddpm_params)


##


fig, ax = plt.subplots(1)
phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))
fig.set_size_inches(*sz)

ax.plot(losses.mean((1, 2, 3)), label="loss")
ax.plot(weighted_losses.mean((1, 2, 3)), label="weighted loss")
ax.set_ylim(0, 20)
ax.legend()
plt.show()
# fig.savefig("losses_simple.svg")


## Compute KL-divergence between q(x_{t-1}|x_t) and p_\phi(x_{t-1}|x_t)

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
    t = jnp.arange(5)

    def kl_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(carry["key"], X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t + np.zeros(num_samples, dtype=np.int32), condition)

        x_t_flat = x_t.reshape((-1, 2048))
        mu_q, Sigma_q = compute_params_forward_posterior(x_t_flat, t, sigma_OU, ddpm_params)

        mu_phi = (1 / jnp.sqrt(ddpm_params["alphas"][t])) * (
                x_t - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * pred
        ).reshape((-1, 2048))

        Sigma_phi = ddpm_params["betas"][t] * jnp.tile(jnp.eye(2048)[None], (num_samples, 1, 1))

        kl = KL_mvn(mu_q, Sigma_q, mu_phi, Sigma_phi)

        carry = {"key": rng}
        return carry, kl

    carry = {"key": rng}
    carry, ys = kl_at_t(carry, t[0])
#    carry, ys = jax.lax.scan(kl_at_t, carry, xs=t)
    return ys


def compute_conditional_kl_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute KL over all diffusion time steps.
    """

    num_samples = len(X)
    t = jnp.arange(5)

    def kl_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(key, X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t + np.zeros(num_samples, dtype=np.int32), condition)

        x_t_flat = x_t.reshape((-1, 2048))
        mu_tilde, Sigma_tilde = compute_params_forward_conditional_posterior(X.reshape((-1, 2048)), x_t_flat, t, ddpm_params)

        mu_phi = (1 / jnp.sqrt(ddpm_params["alphas"][t])) * (
                x_t - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * pred
        ).reshape((-1, 2048))

        Sigma_phi = ddpm_params["betas"][t] * jnp.tile(jnp.eye(2048)[None], (num_samples, 1, 1))

        kl = KL_mvn(mu_tilde, Sigma_tilde, mu_phi, Sigma_phi)

        carry = {"key": rng}
        return carry, kl

    carry = {"key": rng}
    carry, ys = kl_at_t(carry, t[0])
#    carry, ys = jax.lax.scan(kl_at_t, carry, xs=t)
    return ys






ou_params = SimulationParameters(
    sigma_noise=5.,
    tau_x=4,
    tau_y=5,
    C=50,
)


delta_s = jnp.arange(1024)
sigma_OU = compute_ou_temporal_covariance(delta_s, ou_params)[None]
kls_unconditional = compute_unconditional_kl_full_chain(key, state, x_batch, x_batch, sigma_OU, ddpm_params)
kls_conditional = compute_conditional_kl_full_chain(key, state, x_batch, x_batch, ddpm_params)


## Debugging
x = x_batch
x_flat = x.transpose(0,2,1).reshape(-1, 2048)
condition = x_batch
num_samples = len(x)
t = 950
rng, key = jax.random.split(rng)
noise = jax.random.normal(key, x.shape)
x_t = q_sample(x, t, noise, ddpm_params)
pred = state.apply_fn({"params": state.params}, x_t, t + np.zeros(num_samples, dtype=np.int32), condition)

x_t_flat = x_t.transpose(0,2,1).reshape((-1, 2048))
mu_tilde, Sigma_tilde = compute_params_forward_conditional_posterior(x_flat, x_t_flat, t, ddpm_params)
mu_q, Sigma_q = compute_params_forward_posterior(x_t_flat, t, sigma_OU, ddpm_params)

mu_phi = (1 / jnp.sqrt(ddpm_params["alphas"][t])) * (
        x_t - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * pred
).transpose(0,2,1).reshape((-1,2048))

Sigma_phi = ddpm_params["betas"][t] * jnp.tile(jnp.eye(2048)[None], (num_samples, 1, 1))

kl_cond = KL_mvn(mu_tilde, Sigma_tilde, mu_phi, Sigma_phi)
kl_uncond = KL_mvn(mu_q, Sigma_q, mu_phi, Sigma_phi)





