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
from pathlib import Path
from default_config import config as default_config

path = Path(__file__).parent

if "SLURM_JOB_ID" in os.environ:
    base_seed = int(os.environ["SLURM_JOB_ID"])
else:
    base_seed = int(time.time())

print("base seed: ", base_seed, flush=True)

path = Path(__file__).parent

default_config.training.use_full_loss = False
batch_size = 128

config = default_config.copy_and_resolve_references()
config.model.start_filters = 32
config.model.encoder_start_filters = 32
config.data.batch_size = batch_size
config.optim.learning_rate = 2e-3
config.seed = np.random.randint(2 ** 32)
config.training.eval_file = os.path.join(path, "results_simple_loss_32_32", "evals.h5")
config.training.checkpoint_dir = os.path.join(path, "results_simple_loss_32_32", "checkpoint")
config.training.num_train_steps = 100
config.training.num_warmup_steps = 10

train(config)
