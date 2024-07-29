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
    last = X.shape[1] - start - 1  # last possible start index
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
        yield X[i * batch_size : (i + 1) * batch_size], Theta[
            i * batch_size : (i + 1) * batch_size
        ]
        i += 1


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(beta_start, beta_end, timesteps, dtype=jnp.float32)
    return betas


def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps

    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar = jnp.sqrt(1.0 - alphas_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "sqrt_alphas_bar": sqrt_alphas_bar,
        "sqrt_1m_alphas_bar": sqrt_1m_alphas_bar,
    }


def l2_loss(logit, target):
    return (logit - target) ** 2


def simple_loss(params, state, inp, noise):
    x_t, condition, batched_t = inp
    pred = state.apply_fn({"params": params}, x_t, batched_t, condition)

    loss = l2_loss(
        pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
    )

    return loss.mean()


def full_loss(params, state, inp, noise, ddpm_params):
    x_t, condition, batched_t = inp
    pred = state.apply_fn({"params": params}, x_t, batched_t, condition)

    loss = l2_loss(
        pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
    )

    loss = loss / (
        2
        * ddpm_params["alphas"][batched_t, None]
        * (1 - ddpm_params["alphas_bar"][batched_t, None])
    )

    return loss.mean()


def compute_loss_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute loss over all diffusion time steps.
    Jit this function in order to avoid re-compiling inner loss function.
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
        return carry, loss.mean(-1)

    carry = {"key": rng}
    carry, losses = jax.lax.scan(loss_at_t, carry, xs=t)
    return losses


def q_sample(x, t, noise, ddpm_params):
    sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][t, None, None]
    sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][t, None, None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t


def create_train_state(rng, config: ml_collections.ConfigDict):
    """Creates initial `TrainState`."""
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
        normalization=config.model.normalization
    )

    rng, rng_params = jax.random.split(rng, 2)
    input_dims = (1, config.data.length, config.data.channels)
    if config.model.use_encoder:
        condition_dims = input_dims
    else:
        condition_dims = (1, 4)
    params = model.init(
        rng_params,
        jnp.ones(input_dims, dtype=jnp.float32),  # noisy time series
        jnp.ones(input_dims[:1], dtype=jnp.float32),  # t
        jnp.ones(condition_dims, dtype=jnp.float32),  # condition
    )["params"]

    warmup_steps = config.training.num_warmup_steps
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.optim.learning_rate,
        transition_steps=warmup_steps,
    )

    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.optim.learning_rate,
        decay_steps=config.training.num_train_steps - warmup_steps,
    )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps]
    )

    optimizer = optax.adam(
        learning_rate=schedule_fn,
        b1=config.optim.beta1,
        b2=config.optim.beta2,
        eps=config.optim.eps,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    return state


def train_step(rng, state, x, condition, ddpm_params, loss_fn):
    """
    Runs a single training step of the model.

    loss_fn must take the following arguments:
        1. parameters to be differentiated wrt
        2. state object with model
        3. (X, condition, t) - input data for the model
        4. noise - what the model predicts
    """

    print("Tracing train_step", flush=True)

    # run the forward diffusion process to generate noisy image x_t at timestep t

    # create batched timesteps: t with shape (B,)
    B, T, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(
        t_rng, shape=(B,), dtype=jnp.int32, minval=0, maxval=len(ddpm_params["betas"])
    )

    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, (x_t, condition, batched_t), noise)
    metrics = {"loss": loss}
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def norm_data(X, Theta, axis, base_X=None, base_Theta=None):
    """
    Normalize X by the mean and standard deviation of base_X.
    Normalize Theta by the min and max value of base_Theta, such that
    base_Theta is in the interval [-1, 1]
    """
    if base_X is None:
        base_X = X
    if base_Theta is None:
        base_Theta = Theta

    # mean 0, std 1
    X = X - base_X.mean(axis=axis, keepdims=True)
    X /= base_X.std(axis=axis, keepdims=True)

    # limit to [-1, 1]
    Theta = Theta - base_Theta.min(axis=0, keepdims=True)
    Theta /= base_Theta.max(axis=0, keepdims=True)
    Theta *= 2
    Theta -= 1
    return X, Theta


def attrs_to_file(fname, groupname, attrs):
    """
    Save attributes to file ´filename´ in group ´groupname´
    Assume attrs to be a dictionary, and to be a have at
    most 1 level of nesting.
    """

    with h5py.File(fname, "a") as f:
        if groupname in f:
            group = f[groupname]
        else:
            group = f.create_group(groupname)

        for key_outer in attrs.keys():
            config_inner = attrs[key_outer]
            if hasattr(config_inner, "keys"): # if it's another dict, loop through the keys
                for key_inner in config_inner.keys():
                    group.attrs[key_outer + "." + key_inner] = config_inner[key_inner]
            else:
                group.attrs[key_outer] = config_inner


def array_to_file(fname, groupname, dataname, array):
    """
    Save array to file ´filename´ in group ´groupname´ with name ´dataname´
    """
    with h5py.File(fname, "a") as f:
        if groupname is None:
            group = f
        else:
            if groupname in f:
                group = f[groupname]
            else:
                group = f.create_group(groupname)
        group.create_dataset(dataname, data=array)


def run_to_file(fn, fname, dataname, step, *args):
    results = fn(*args)
    grp = f"{step}"
    array_to_file(fname, grp, dataname, results)


def train(config):
    rng = jax.random.PRNGKey(config.seed)

    ## Load data
    X = np.load(config.data.X_train_path)
    Theta = np.load(config.data.Theta_train_path)
    X, Theta = norm_data(X, Theta, axis=config.data.norm_axis)

    X_fp = np.load(config.data.X_fixed_points_path)
    X_fp = X_fp[:, 1024:2048, :]
    Theta_fp = np.load(config.data.Theta_fixed_points_path)
    X_fp, Theta_fp = norm_data(X_fp, Theta_fp, axis=config.data.norm_axis)


    train_gen = partial(
        train_data_gen, batch_size=config.data.batch_size, X=X, Theta=Theta
    )

    ## Create data generator, init model
    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)

    ## Jit and partial
    ddpm_params = get_ddpm_params(config.ddpm)
    loss_fn = partial(full_loss, ddpm_params=ddpm_params) if config.training.use_full_loss else simple_loss
    train_step_jit = partial(train_step, ddpm_params=ddpm_params, loss_fn=loss_fn)
    train_step_jit = jax.jit(train_step_jit)

    compute_loss_full_chain_jit = partial(
        compute_loss_full_chain, X=X_fp, condition=X_fp, ddpm_params=ddpm_params
    )
    compute_loss_full_chain_jit = jax.jit(compute_loss_full_chain_jit)

    sample_step_jit = partial(ddpm_sample_step, ddpm_params=ddpm_params)
    sample_step_jit = jax.jit(sample_step_jit)
    sample_loop_partial = partial(
        sample_loop,
        shape=X_fp.shape,
        condition=X_fp,
        sample_step=sample_step_jit,
        timesteps=config.ddpm.timesteps,
    )


    # Save conditional values for sampling to file
    array_to_file(config.training.eval_file, None, "conditions", X_fp)
    attrs_to_file(config.training.eval_file, "config", config)


    ## Init helper vars and training loop
    step = 0

    start_time = time.time()


    # Log initial losses        
    rng, key = jax.random.split(rng)
    run_to_file(
        compute_loss_full_chain_jit,
        config.training.eval_file,
        "full_chain_loss",
        step,
        key,
        state,
    )

    # Training loop
    break_flag = False
    eval_flag = False
    while True:
        rng, key = jax.random.split(rng)
        train_iter = train_gen(key)
        for batch in train_iter:
            tic = time.time()
            print(step, end="\r", flush=True)
            rng, train_step_rng = jax.random.split(rng, 2)
            condition = batch[0] if config.model.use_encoder else batch[1]
            state, metrics = train_step_jit(train_step_rng, state, batch[0], condition)
            toc = time.time()
            step += 1
            if step > config.training.num_train_steps:
                break_flag = True
                eval_flag = True
                break

            if step % config.training.eval_every == 0:
                eval_flag = True

        if eval_flag:
            rng, key = jax.random.split(rng)
            run_to_file(
                compute_loss_full_chain_jit,
                config.training.eval_file,
                "full_chain_loss",
                step,
                key,
                state,
            )

            rng, key = jax.random.split(rng)
            run_to_file(
                sample_loop_partial,
                config.training.eval_file,
                "samples",
                step,
                key,
                state,
            )
            eval_flag = False

        if break_flag:
            ckpt = {'model': state, 'config': config}
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(config.training.checkpoint_dir, ckpt, save_args=save_args)
            break

