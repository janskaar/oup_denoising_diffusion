import jax
import jax.numpy as jnp
import orbax
import orbax.checkpoint
import optax
import flax.linen as nn
from flax.training import train_state, orbax_utils
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.train.train_utils import step
from flowjax.train.losses import MaximumLikelihoodLoss
from flowjax import wrappers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os, itertools, pickle, h5py
from scipy.signal import welch
from functools import partial
import equinox as eqx

from ou_diffusion_funcs import sample_ou_process, sample_prior_and_ou_process, OUParams
from unet import Encoder



encoder_dir = "results_latent_4_global_norm"
run_id = "26"
ckpt_path = os.path.join(encoder_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
encoder_params = raw_restored["model"]["params"]["Encoder_0"]

def sample_and_encode(keys, prior_min, prior_max, norm_scale):
    x, theta = sample_prior_and_ou_process(keys, prior_min, prior_max)
    x = x / norm_scale
    enc = Encoder(start_filters = 16,
                filter_mults = (1, 2, 4, 8),
                latent_dim = 4,
                normalization = True,
                activation = nn.silu).apply({"params": encoder_params}, x)
    return enc, theta


def fit_to_data(
    key,
    dist,
    max_patience,
    max_steps,
    batch_size,
    optimizer,
):

    loss_fn = MaximumLikelihoodLoss()

    params, static = eqx.partition(
        dist,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, wrappers.NonTrainable),
    )
    best_params = params
    opt_state = optimizer.init(params)

    rng, key = jax.random.split(key)
    losses = []

    loop = tqdm(range(max_steps))

    minloss = 1e10
    num_fruitless = 0
    for _ in loop:
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, batch_size)
        batch = sample_and_encode(keys, np.ones(4) * 1., jnp.ones(4) * 10., 1.743)
        return batch
        key, subkey = jax.random.split(key)
        params, opt_state, loss_i = step(
            params,
            static,
            *batch,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            key=subkey,
        )
        losses.append(loss_i)
        if losses[-1] < minloss:
            minloss = losses[-1]
            num_fruitless = 0
        else:
            num_fruitless += 1
    
        if num_fruitless > max_patience:
            loop.set_postfix_str(f"{loop.postfix} (Max patience reached)")
            break

    dist = eqx.combine(params, static)
    return dist, losses

rng = jax.random.PRNGKey(1234)
rng, key = jax.random.split(rng)



flow = masked_autoregressive_flow(
    key=key,
    base_dist=Normal(jnp.zeros(4)),
    cond_dim=4,
    flow_layers=1,
)


rng, key = jax.random.split(rng)
opt = optax.adam(1e-10)
flow, losses = fit_to_data(
    key=key,
    dist=flow,
    max_patience=5,
    max_steps=10,
    batch_size=5,
    optimizer=opt
)


