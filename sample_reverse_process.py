import os, time, h5py
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import orbax.checkpoint

from sampling import sample_loop, ddpm_sample_step
from diffusion import norm_data
from unet import Encoder

from simulator import ParticleSimulator, SimulationParameters

top_dir = "results_reduced_parameterspace_norm_012_latent_4"
run_id = "47"
ckpt_path = os.path.join(top_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
diffusion_params = raw_restored["model"]["params"]

train_X = np.load(os.path.join("data", "z_reduced_parameterspace.npy"))
train_Theta = np.load(os.path.join("data", "theta_reduced_parameterspace.npy"))


# Simulate 5 different simulations with same seed, use the 5 resulting
# time series to sample reverse diffusion 100 times each.
sim_params = np.array([4., 6., 2., 7.])
num_procs = 5
z0 = np.zeros((num_procs, 2))

## Simulations
params = SimulationParameters(
    num_procs=num_procs,
    sigma_noise=sim_params[0],
    tau_x=sim_params[1],
    tau_y=sim_params[2],
    C=sim_params[3],
)

simulator = ParticleSimulator(z0, params)
simulator.simulate(2048)
z = simulator.z
z = z[::10, :][:,-1024]

# Condition for reverse process
x = np.tile(z[None], (100, 1, 1))
x, _ = norm_data(x, train_Theta, base_X=train_X, axis=(0,1,2))


## Sample

ddpm_params = get_ddpm_params(config.ddpm)

sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
sample_step = jax.jit(sample_step)

sample = sample_loop(
     key, state, x[:5].shape, x[:5], sample_step, 1#config.ddpm.timesteps
)


## Save

with h5py.File(os.path.join(top_dir, f"samples_reverse_process_run_{run_id}_params_4_6_2_7.h5"), "w") as f:
    f.create_dataset("samples", data=np.asarray(sample))
    f.create_dataset("conditions", data=x)



