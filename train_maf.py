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
from pathlib import Path
from scipy.signal import welch
from scipy.stats.qmc import Halton
from functools import partial
import equinox as eqx

from ou_diffusion_funcs import sample_ou_process, sample_prior_and_ou_process, OUParams
from unet import Encoder

path = Path(__file__).parent

# For normalizing marginals
OUP_SCALE = 1.743 # global

# per latent dimension
ENC_MEAN = np.array([[1008, 387, 710, 754]])
ENC_SCALE = np.array([[177, 208, 365, 167]])

encoder_dir = "results_latent_4_global_norm"
run_id = "26"
ckpt_path = os.path.join(encoder_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
encoder_params = raw_restored["model"]["params"]["Encoder_0"]

@jax.jit
def sample_and_encode(keys, prior_min, prior_max, norm_scale):
    x, theta = sample_prior_and_ou_process(keys, prior_min, prior_max)
    x = x / norm_scale
    enc = Encoder(start_filters = 16,
                filter_mults = (1, 2, 4, 8),
                latent_dim = 4,
                normalization = True,
                activation = nn.silu).apply({"params": encoder_params}, x)
    enc = enc - ENC_MEAN
    enc = enc / ENC_SCALE
    return enc, theta


@jax.jit
def sample_x_and_encode(keys, theta, norm_scale):
    params = OUParams(sigma2_noise = theta[0],
                      tau_x = theta[1],
                      tau_y = theta[2],
                      c = theta[3])

    x = jax.vmap(sample_ou_process, in_axes=(0, None))(keys, params)
    x = x / norm_scale
    enc = Encoder(start_filters = 16,
                filter_mults = (1, 2, 4, 8),
                latent_dim = 4,
                normalization = True,
                activation = nn.silu).apply({"params": encoder_params}, x)
    enc = enc - ENC_MEAN
    enc = enc / ENC_SCALE
    return enc


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
        batch = sample_and_encode(keys, np.ones(4) * 1., jnp.ones(4) * 10., OUP_SCALE)
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

out_dir = os.path.join(path, "maf_results", "2_layers_200_hidden_norm_encodings")

sample_thetas = [np.array([2., 2., 2., 2.]),
                 np.array([2., 9., 2., 9.]),
                 np.array([9., 2., 9., 2.]),
                 np.array([5., 5., 5., 5.]),
                 np.array([2., 2., 9., 9.]),
                 np.array([9., 9., 2., 2.]),
                 np.array([9., 9., 9., 9.])]

num_samples = 200

num_runs = 20

sampler = Halton(d=1, scramble=True, seed=np.random.randint(2 ** 32))
learning_rates = sampler.random(n=num_runs)

# log space
learning_rates *= (np.log(1) - np.log(1e-5))
learning_rates += np.log(1e-5)
learning_rates = np.exp(learning_rates).squeeze()
 
print("Learning rates")
print(learning_rates)

losses_outer = []
for i in range(num_runs):
    flow = masked_autoregressive_flow(
        key=key,
        base_dist=Normal(jnp.zeros(4)),
        cond_dim=4,
        flow_layers=2,
        nn_width=200
    )

    rng, key = jax.random.split(rng)
    opt = optax.adam(learning_rates[i])
    flow, losses = fit_to_data(
        key=key,
        dist=flow,
        max_patience=200,
        max_steps=5000,
        batch_size=200,
        optimizer=opt
    )

    losses = np.array(losses)

    flow_samples = []
    encoded_samples = []
    for j, theta in enumerate(sample_thetas):
        rng, key = jax.random.split(rng)
        flow_samples.append(flow.sample(key, condition=theta, sample_shape=(num_samples,)))
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, num_samples)
        encoded_samples.append(sample_x_and_encode(keys, theta, OUP_SCALE))

    print(losses)
    with h5py.File(os.path.join(out_dir, "save.h5"), "a") as f:
        grp = f.create_group(str(i))
        grp.attrs["learning_rate"] = learning_rates[i]
        grp.create_dataset("loss", data=losses)
        grp.create_dataset("flow_sample", data=np.array(flow_samples))
        grp.create_dataset("encoded_sample", data=np.array(encoded_samples))

    eqx.tree_serialise_leaves(os.path.join(out_dir, f"flow_{i}.eqx"), flow)

    print(f"Run {i}, minloss = {losses.min()}")

