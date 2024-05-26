from simulator import ParticleSimulator, SimulationParameters
from unet import UNET
from diffusion import train, get_ddpm_params
from sampling import sample_loop, ddpm_sample_step
import jax
import numpy as np
from scipy.signal import welch
import os, time
from sklearn.preprocessing import StandardScaler
from functools import partial
import ml_collections
import matplotlib.pyplot as plt

## Config for diffusion model
config = ml_collections.ConfigDict()

# training
config.training = training = ml_collections.ConfigDict()
training.num_train_steps = 20

# ddpm 
config.ddpm = ddpm = ml_collections.ConfigDict()
ddpm.beta_schedule = "linear"
ddpm.timesteps = 1000

# data
config.data = data = ml_collections.ConfigDict()
data.batch_size = 8
data.length = 1024
data.channels = 2

# model
config.model = model = ml_collections.ConfigDict()
model.use_encoder = True
model.start_filters = 16
model.filter_mults = (1, 2, 4, 8)
model.encoder_start_filters = 16
model.encoder_filter_mults = (1, 2, 4, 8)
model.encoder_latent_dim = 4
model.use_attention = False

# optim
config.optim = optim = ml_collections.ConfigDict()
optim.optimizer = 'Adam'
optim.learning_rate = 1e-3
optim.beta1 = 0.9
optim.beta2 = 0.999
optim.eps = 1e-8
optim.warmup_steps = 5 

config.seed = 123

X = np.load(os.path.join("data", "z.npy"))
Theta = np.load(os.path.join("data", "theta.npy"))

losses, state = train(config, X, Theta)

ddpm_params = get_ddpm_params(config.ddpm)
sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
sample_step = jax.jit(sample_step)
sample_shape = (20, 1024, 2)

condition = X[:20,1024:2048]

rng = jax.random.PRNGKey(1)
rng, key = jax.random.split(rng)
sample = sample_loop(key, state, sample_shape, condition, sample_step, config.ddpm.timesteps)

fs, sample_psd = welch(sample.squeeze())
fs, data_psd = welch(X.squeeze())

plt.plot(sample_psd.mean(0), label="sample", color="black")
plt.plot(data_psd.mean(0), label="data full")
plt.plot(data_psd_1.mean(0), label="data 1")
plt.plot(data_psd_2.mean(0), label="data 2")
plt.legend()
plt.show()

