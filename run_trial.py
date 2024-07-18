from simulator import ParticleSimulator, SimulationParameters
from unet import UNET
from diffusion import train, get_ddpm_params
from sampling import sample_loop, ddpm_sample_step
import jax
import numpy as np
from scipy.signal import welch
from scipy.stats.qmc import Halton
import os, time, h5py, re
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from default_config import config as default_config

if "SLURM_PROCID" in os.environ:
    base_seed = int(os.environ["SLURM_JOB_ID"])
else:
    base_seed = int(time.time())

path = Path(__file__).parent
print("base seed: ", base_seed, flush=True)

np.random.seed(base_seed)

batch_size = 128
default_config.training.use_full_loss = False
default_config.model.start_filters = 32
default_config.model.encoder_start_filters = 16
default_config.model.use_encoder = False
default_config.model.use_parameters = False
default_config.data.X_train_path = os.path.join(path, "data", "z.npy")
default_config.data.X_fixed_points_path = os.path.join(path, "data", "z_fixed_points.npy")
default_config.data.Theta_train_path = os.path.join(path, "data", "theta.npy")
default_config.data.Theta_fixed_points_path = os.path.join(path, "data", "theta_fixed_points.npy")
default_config.data.norm_axis = (0,1,2)
default_config.data.batch_size = batch_size
default_config.training.num_train_steps = 5000
default_config.training.num_warmup_steps = 500

outdir = os.path.join(path, "results_simple_loss_norm_012")

num_samples = 10


sampler = Halton(d=1, scramble=True, seed=np.random.randint(2 ** 32))
learning_rates = sampler.random(n=10)

# log space
learning_rates *= (np.log(1e-1) - np.log(1e-5))
learning_rates += np.log(1e-5)
learning_rates = np.exp(learning_rates).squeeze()





for i, lr in enumerate(learning_rates):
    config = default_config.copy_and_resolve_references()
    config.optim.learning_rate = lr
    config.seed = np.random.randint(2 ** 32)
    config.training.eval_file = os.path.join(outdir, f"run_{i}.h5")
    config.training.checkpoint_dir = os.path.join(outdir, f"checkpoint_{i}")
    train(config)

