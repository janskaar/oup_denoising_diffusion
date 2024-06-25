from simulator import ParticleSimulator, SimulationParameters
from unet import UNET
from diffusion import train, get_ddpm_params
from sampling import sample_loop, ddpm_sample_step
import jax
import numpy as np
from scipy.signal import welch
import os, time, h5py, re
from functools import partial
import matplotlib.pyplot as plt
from default_config import config as default_config


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

def train_and_save(config):
    tic = time.time()
    metrics, state = train(config, X, Theta)
    toc = time.time()

    print(f"Trained in {toc - tic:.1f} seconds", flush=True)

    rng = jax.random.PRNGKey(1)
    rng, key = jax.random.split(rng)
    sample = sample_loop(
        key, state, sample_shape, condition, sample_step, config.ddpm.timesteps
    )


    # lr representation
    base, exp = (int(d) for d in re.findall("\d+", f"{config.optim.learning_rate:.0E}"))
    with h5py.File(f"results/results_simple_loss_b{config.data.batch_size}_lr{base}_{exp}.h5", "w") as f:
        f.create_dataset("sample", data=sample)
        f.create_dataset("condition", data=condition)
        f.create_dataset("loss", data=metrics["loss"])
        f.create_dataset("time", data=metrics["step_time"])

        grp = f.create_group("config")
        for key_outer in config.keys():
            config_inner = config[key_outer]
            if hasattr(config_inner, "keys"): # if it's another dict, loop through the keys
                for key_inner in config_inner.keys():
                    grp.attrs[key_outer + "." + key_inner] = config_inner[key_inner]
            else:
                grp.attrs[key_outer] = config_inner

batch_sizes = [64, 128, 512]
lrs = [1e-3, 1e-4, 1e-5, 1e-6]

for batch_size in batch_sizes:
    for lr in lrs:
        config = default_config.copy_and_resolve_references()
        config.data.batch_size = batch_size
        config.optim.learning_rate = lr
        train_and_save(config)
        break
    break









