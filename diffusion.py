# simplified form of https://github.com/yiyixuxu/denoising-diffusion-flax

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, orbax_utils
from flax import jax_utils

import optax
import orbax.checkpoint
import numpy as np
import ml_collections

import os, re, h5py, time, json
from functools import partial
from itertools import product
import matplotlib.pyplot as plt
from scipy.signal import welch

from sampling import sample_loop, ddpm_sample_step, model_predict
from unet import UNET, SinusoidalPosEmb


def float_to_str(x):
    return f"{x:.3f}"


def write_log(step, loss, theta, fname):
    if not os.path.isfile(fname):
        with open(fname, "w") as f:
            pass

    with open(fname, "a") as f:
        f.write(f"{step}," + ",".join(map(float_to_str, loss)) + "\n") 


def random_slice(X, start, key, length):
    """
    X: (N, L, C) tensor, will slice along axis L
    start: int, first possible index to select
    length: int, length of slice 

    For jitting, partial out length first
    """

    @partial(jax.vmap, in_axes=(0, 0, None))
    def slice(X, start, length):
        return jax.lax.dynamic_slice_in_dim(X, start, length)

    num = len(X)
    last = X.shape[1] - start - 1 # last possible start index
    start_indices = jax.random.randint(key, shape=(num,), minval=start, maxval=last) 
    return slice(X, start_indices, length)

random_slice_jit = jax.jit(partial(random_slice, length=1024))

def train_data_gen(key, batch_size, X, Theta):
    rng, key = jax.random.split(key)
    X = jax.random.permutation(key, X)
    rng, key = jax.random.split(rng)
    X = random_slice_jit(X, 0, key)
    i = 0

    num = len(X) // batch_size
    if len(X) % batch_size != 0:
        num += 1

    while i <= num - 1:
        yield X[i * batch_size : (i + 1) * batch_size], Theta[i * batch_size : (i + 1) * batch_size]
        i += 1


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(
        beta_start, beta_end, timesteps, dtype=jnp.float32)
    return betas


def cosine_beta_schedule(timesteps):
    """Return cosine schedule 
    as proposed in https://arxiv.org/abs/2102.09672 """
    s=0.008
    max_beta=0.999
    ts = jnp.linspace(0, 1, timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return(jnp.clip(betas, 0, max_beta))


def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps

    if schedule_name == "linear":
        betas = linear_beta_schedule(timesteps)
    elif schedule_name == "cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"unknown beta schedule {schedule_name}")
    assert betas.shape == (timesteps,)
    alphas = 1. - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar= jnp.sqrt(1. - alphas_bar)

    return {
      "betas": betas,
      "alphas": alphas,
      "alphas_bar": alphas_bar,
      "sqrt_alphas_bar": sqrt_alphas_bar,
      "sqrt_1m_alphas_bar": sqrt_1m_alphas_bar
  }


def l2_loss(logit, target):
    return (logit - target)**2


def q_sample(x, t, noise, ddpm_params):

    sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][t, None, None]
    sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][t, None, None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t


def create_train_state(rng, config: ml_collections.ConfigDict):
    """Creates initial `TrainState`."""
    model = UNET(
        start_filters = config.model.start_filters, 
        filter_mults = config.model.filter_mults,
        out_channels = config.data.channels,
        activation = nn.silu,
        encoder_start_filters = config.model.encoder_start_filters,
        encoder_filter_mults = config.model.encoder_filter_mults,
        encoder_latent_dim = config.model.encoder_latent_dim,
        use_encoder = config.model.use_encoder,
        attention = config.model.use_attention
    )

    rng, rng_params = jax.random.split(rng, 2)
    input_dims = (1, config.data.length, config.data.channels)
    if config.model.use_encoder:
        condition_dims = input_dims
    else:
        condition_dims = (1, 4)
    params = model.init(
        rng_params, 
        jnp.ones(input_dims, dtype=jnp.float32), # noisy time series
        jnp.ones(input_dims[:1], dtype=jnp.float32), # t
        jnp.ones(condition_dims, dtype=jnp.float32), # condition
    )["params"]


    warmup_steps = config.optim.warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=config.optim.learning_rate,
        transition_steps=warmup_steps)

    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.optim.learning_rate,
        decay_steps=config.training.num_train_steps - warmup_steps)

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps])

    optimizer = optax.adam(
        learning_rate = schedule_fn,
        b1=config.optim.beta1,
        b2=config.optim.beta2,
        eps=config.optim.eps
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optimizer, 
    )

    return state


def train_step(rng, state, batch, ddpm_params, use_encoder=True):
    print("Tracing train_step", flush=True)

    # run the forward diffusion process to generate noisy image x_t at timestep t
    x = batch[0]
    theta = batch[1]

    # create batched timesteps: t with shape (B,)
    B, T, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params["betas"]))

    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)

    def compute_loss(params):
        if use_encoder:
            pred = state.apply_fn({"params":params}, x_t, batched_t, x)
        else:
            pred = state.apply_fn({"params":params}, x_t, batched_t, theta)

        loss = l2_loss(
                pred.reshape((pred.shape[0], -1)), 
                noise.reshape((noise.shape[0], -1))
                )
        loss = jnp.mean(loss, axis= 1)
        return loss.mean()

    # set this as static_argnum, and compile one version that only returns loss values,
    # and one version that does the entire train step
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    metrics = {"loss": loss}
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def train(config, X, Theta):

    rng = jax.random.PRNGKey(config.seed)

    train_gen = partial(train_data_gen, batch_size=config.data.batch_size, X=X, Theta=Theta)
    num_steps = config.training.num_train_steps
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)

    ddpm_params = get_ddpm_params(config.ddpm)

    train_step_jit = partial(train_step, ddpm_params=ddpm_params)
    train_step_jit = jax.jit(train_step_jit)

#     sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
#     sample_step = jax.jit(sample_step)
    step = 0

    start_time = time.time()

    # for plotting loss
    loss_history = []
    break_flag = False
    while True:
        rng, key = jax.random.split(rng) 
        train_iter = train_gen(key)

        tic = time.time()
        for batch in train_iter:
            print(len(batch))
            print(batch[0].shape)
            print(batch[1].shape)

            print(step, end="\r", flush=True)
            rng, train_step_rng = jax.random.split(rng, 2)
            state, metrics = train_step_jit(train_step_rng, state, batch)
            loss_history.append(metrics["loss"])
            step += 1
            if step > config.training.num_train_steps:
                break_flag = True
                break

        if break_flag:
            break

    return np.array(loss_history), state

