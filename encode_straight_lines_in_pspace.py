import os, time, h5py
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import orbax.checkpoint

from diffusion import norm_data
from unet import Encoder

from simulator import ParticleSimulator, SimulationParameters


top_dir = "results_reduced_parameterspace_norm_012_latent_5"
run_id = "0"
ckpt_path = os.path.join(top_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
encoder_params = raw_restored["model"]["params"]["Encoder_0"]

X_path = os.path.join("data", "z_reduced_parameterspace.npy")
X = np.load(X_path)
Theta_path = os.path.join("data", "theta_reduced_parameterspace.npy")
Theta = np.load(Theta_path)
X, Theta = norm_data(X, Theta, axis=(0,1,2))


def sim_and_encode(param_vals, num):
    """
    Run ´num´ simulations with parameters ´param_vals´, and run the
    encoder on the resulting time series.

    The encoder is run on multiple slices on the time series, at
    10 ms separation between slices.
    """

    z0 = np.zeros((num, 2))
    zs = []

    ## Simulations
    params = SimulationParameters(
        num_procs=num,
        sigma_noise=param_vals[i,0],
        tau_x=param_vals[i,1],
        tau_y=param_vals[i,2],
        C=param_vals[i,3],
    )

    simulator = ParticleSimulator(z0, params)
    simulator.simulate(3000)
    zs.append(simulator.z.squeeze())

    zs = np.array(zs)

    zs = zs[:, :30000:10, :]

    x = zs[:,1024:]
    start_indices = jnp.arange(1024, 1624+1, 10)
    encs = []
    for start in start_indices:
        enc = Encoder(start_filters = 16,
                     filter_mults = (1, 2, 4, 8),
                     latent_dim = 5,
                     normalization = True,
                     activation = nn.silu).apply({"params": encoder_params}, x)
        encs.append(enc)  
    encs = np.array(encs)
    return x, encs

center = np.array([[3, 7.5, 3, 7.5]])
line = center + np.linspace(-2, 2, 21)[:,None,None,None]

x, enc = sim_and_encode(line, 100)

