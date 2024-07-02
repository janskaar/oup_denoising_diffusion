from simulator import ParticleSimulator, SimulationParameters
from unet import UNET
from diffusion import train, get_ddpm_params
from sampling import sample_loop, ddpm_sample_step
import jax
from flax.training import orbax_utils
from flax import jax_utils
import orbax.checkpoint
import numpy as np
from scipy.signal import welch
from scipy.stats.qmc import Halton
import os, time, h5py, re
from functools import partial
import matplotlib.pyplot as plt
from default_config import config as default_config

if "SLURM_JOB_ID" in os.environ:
    base_seed = int(os.environ["SLURM_JOB_ID"])
else:
    base_seed = int(time.time())

print("base seed: ", base_seed, flush=True)

np.random.seed(base_seed)

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

X_fp = np.load(os.path.join("data", "z_fixed_points.npy"))
Theta_fp = np.load(os.path.join("data", "theta_fixed_points.npy"))
X_fp, Theta_fp = norm_data(X_fp, Theta_fp)
 
# ddpm params are the same for all configs
ddpm_params = get_ddpm_params(default_config.ddpm)
sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
sample_step = jax.jit(sample_step)

sample_shape = (len(X_fp), 1024, 2)
condition = X_fp[:, 1024:2048]

def train_and_save(config, sim_index):
    tic = time.time()
    metrics, state = train(config, X, Theta)
    toc = time.time()

    print(f"Trained in {toc - tic:.1f} seconds", flush=True)

    rng = jax.random.PRNGKey(config.seed)
    rng, key = jax.random.split(rng)
    sample = sample_loop(
        key, state, sample_shape, condition, sample_step, config.ddpm.timesteps
    )


    with h5py.File(outfile, "a") as f:
        grp = f.create_group(f"{sim_index:02d}")
        grp.create_dataset("sample", data=sample)
        grp.create_dataset("condition", data=condition)
        grp.create_dataset("loss", data=metrics["loss"])
        grp.create_dataset("time", data=metrics["step_time"])

        subgrp = grp.create_group("config")
        for key_outer in config.keys():
            config_inner = config[key_outer]
            if hasattr(config_inner, "keys"): # if it's another dict, loop through the keys
                for key_inner in config_inner.keys():
                    subgrp.attrs[key_outer + "." + key_inner] = config_inner[key_inner]
            else:
                subgrp.attrs[key_outer] = config_inner

    ckpt = {'model': state, 'config': config}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(model_outfile, ckpt, save_args=save_args)


default_config.optim.use_full_loss = False
outfile = f"results_final/results_simple_loss_32_32.h5"
model_outfile = f"results_final/checkpoint"

batch_size = 128

config = default_config.copy_and_resolve_references()
config.model.start_filters = 32
config.model.encoder_start_filters = 32
config.data.batch_size = batch_size
config.optim.learning_rate = lr
config.seed = np.random.randint(2 ** 32)
config.training.num_train_steps = 100
train_and_save(config, 1)

