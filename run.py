from simulator import ParticleSimulator, SimulationParameters
from unet import UnconditionalUNET as UNET
from diffusion import train, get_ddpm_params
from sampling import sample_loop, ddpm_sample_step
import jax
import numpy as np
from scipy.signal import welch
import os, time
from functools import partial
import ml_collections
import matplotlib.pyplot as plt

## Config for diffusion model
config = ml_collections.ConfigDict()

# training
config.training = training = ml_collections.ConfigDict()
training.num_train_steps = 1000

# ddpm 
config.ddpm = ddpm = ml_collections.ConfigDict()
ddpm.beta_schedule = "linear"
ddpm.timesteps = 1000

# data
config.data = data = ml_collections.ConfigDict()
data.batch_size = 8
data.length = 1024
data.channels = 1

# model
config.model = model = ml_collections.ConfigDict()
model.unet = "UnconditionalUNET"
model.start_filters = 16
model.filter_mults = (1, 2, 4, 8)

# optim
config.optim = optim = ml_collections.ConfigDict()
optim.optimizer = 'Adam'
optim.learning_rate = 1e-3
optim.beta1 = 0.9
optim.beta2 = 0.999
optim.eps = 1e-8
optim.warmup_steps = 200

config.seed = 123

## Simulations
params1 = SimulationParameters(num_procs=5000)
z0 = np.zeros((params1.num_procs, 2))
simulator1 = ParticleSimulator(z0, params1)
simulator1.simulate(500)

params2 = SimulationParameters(num_procs=5000, tau_x=5., C=100, tau_y=0.5)
z0 = np.zeros((params2.num_procs, 2))
simulator2 = ParticleSimulator(z0, params2)
simulator2.simulate(500)

X = np.concatenate((simulator1.z[500:,:,1:2].transpose((1, 0, 2)), simulator2.z[500:,:,1:2].transpose((1, 0, 2))), axis=0)
X -= X.mean(axis=1, keepdims=True)
X /= X.std(axis=1, keepdims=True)
X_shuffle = X.copy()
np.random.shuffle(X_shuffle)
losses, state = train(config, X_shuffle)

ddpm_params = get_ddpm_params(config.ddpm)
sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
sample_step = jax.jit(sample_step)
sample_shape = (20, 1024, 1)
x = X[:20,1024:2048]
z = x * np.sqrt(ddpm_params["alphas_bar"][-1]) + np.random.randn(*x.shape) * np.sqrt(1 - ddpm_params["alphas_bar"][-1])

rng = jax.random.PRNGKey(1)
rng, key = jax.random.split(rng)
sample = sample_loop(key, state, sample_shape, sample_step, config.ddpm.timesteps, z=None)

fs, sample_psd = welch(sample.squeeze())
fs, data_psd = welch(X.squeeze())
data_psd_1 = data_psd[:5000]
data_psd_2 = data_psd[5000:]

plt.plot(sample_psd.mean(0), label="sample", color="black")
plt.plot(data_psd.mean(0), label="data full")
plt.plot(data_psd_1.mean(0), label="data 1")
plt.plot(data_psd_2.mean(0), label="data 2")
plt.legend()
plt.show()

