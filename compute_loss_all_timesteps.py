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
from diffusion import get_ddpm_params
from unet import UNET
from default_config import config
from sampling import ddpm_sample_step
from diffusion import l2_loss, q_sample, simple_loss

f = os.path.join("results_simple", "checkpoint")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(f)
params = raw_restored["model"]["params"]

model = UNET(
    start_filters=config.model.start_filters,
    filter_mults=config.model.filter_mults,
    out_channels=config.data.channels,
    activation=nn.silu,
    encoder_start_filters=config.model.encoder_start_filters,
    encoder_filter_mults=config.model.encoder_filter_mults,
    encoder_latent_dim=config.model.encoder_latent_dim,
    use_encoder=config.model.use_encoder,
    use_parameters=config.model.use_parameters,
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


def sample_loop(rng, state, shape, condition, sample_step, timesteps):
    rng, x_rng = jax.random.split(rng)
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    print("Shape: ", shape)
    # sample step
    print("Sampling")
    xs = []
    x_means = []
    for t in reversed(jnp.arange(timesteps)):
        print(t, end="\r")
        rng, step_rng = jax.random.split(rng)
        # x_mean is final prediction at t=0
        x, x_mean = sample_step(state, step_rng, x, t, condition)
        xs.append(x)
        x_means.append(x_mean)

    return np.array(xs), np.array(x_means)


ddpm_params = get_ddpm_params(config.ddpm)
sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
sample_step = jax.jit(sample_step)

rng = jax.random.PRNGKey(1)
rng, key = jax.random.split(rng)

x_batch = X[:20, 1024:2048]

# sample_shape = (20, 1024, 2)
# condition = X[:20, 1024:2048]

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

# losses = []
# for i in reversed(range(1000)):
#     print(i, end="\r")
#     rng, key = jax.random.split(rng)
#     loss, pred, x_t = loss_step(key, state, x_batch, i, ddpm_params)
#     losses.append(loss) 
# losses = np.array(losses)
# 
# weighted_losses = losses / (
#     2
#     * ddpm_params["alphas"][:,None]
#     * (1 - ddpm_params["alphas_bar"])[:,None]
# )

##


def compute_loss_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute loss over all diffusion time steps.
    """

    t = jnp.tile(jnp.arange(1000)[::-1][:,None], (1, len(X)))

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
        return carry, loss.mean(-1)

    carry = {"key": rng}
    carry, losses = jax.lax.scan(loss_at_t, carry, xs=t)
    return losses


rng, key = jax.random.split(rng)
losses = compute_loss_full_chain(key, state, x_batch, x_batch, ddpm_params)
compute_loss_full_chain_jit = jax.jit(compute_loss_full_chain)
losses = compute_loss_full_chain_jit(key, state, x_batch, x_batch, ddpm_params)


##

fig, ax = plt.subplots(1)
phi = np.arctan(1080 / 1920)
sz = (14 * np.cos(phi), 14 * np.sin(phi))
fig.set_size_inches(*sz)

ax.plot(losses.mean(1), label="loss")
ax.plot(weighted_losses.mean(1), label="weighted loss")
ax.set_ylim(0, 20)
ax.legend()
plt.show()
# fig.savefig("losses_simple.svg")

