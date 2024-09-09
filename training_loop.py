import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, orbax_utils

import optax
import orbax.checkpoint
import numpy as np
import ml_collections

import os, re, h5py, time, json
from functools import partial

from sampling import sample_loop, ddpm_sample_step
from unet import UNET, SinusoidalPosEmb
from diffusion import (linear_beta_schedule,
                      get_ddpm_params,
                      simple_loss,
                      q_sample,
                      compute_loss_full_chain)

from ou_diffusion_funcs import sample_ou_process, sample_prior_and_ou_process, OUParams


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

    prior_min = np.array(config.data.prior_min)
    prior_max = np.array(config.data.prior_max)

    ## Create fixed point data
    rng, key = jax.random.split(rng)
    Theta_fp = jax.random.uniform(key, minval=prior_min, maxval=prior_max, shape=(9, 4))
    Theta_fp = jnp.tile(Theta_fp[:,None,:], (1, 50, 1)).reshape((-1 ,4))
    ouparams = OUParams(sigma2_noise=Theta_fp[:,0],
                        tau_x=Theta_fp[:,1],
                        tau_y=Theta_fp[:,2],
                        c=Theta_fp[:,3])
    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, 450)
    X_fp = jax.vmap(sample_ou_process)(keys, ouparams)
    X_fp = X_fp - config.data.norm_shift
    X_fp = X_fp / config.data.norm_scale

    ## Init model
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
        keys = jax.random.split(key, config.data.batch_size)
        batch = sample_prior_and_ou_process(keys, prior_min, prior_max)
        x = batch[0]
        x = x - config.data.norm_shift
        x = x / config.data.norm_scale

        condition = x if config.model.use_encoder else batch[1]
        rng, key = jax.random.split(rng)
        if config.training.dump_batch: # for testing only
            return x, batch[1]
        state, metrics = train_step_jit(rng, state, x, condition)
        step += 1
        if step > config.training.num_train_steps:
            break_flag = True
            eval_flag = True

        if step % config.training.eval_every == 0:
            print(f"Step {step}", flush=True)
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

