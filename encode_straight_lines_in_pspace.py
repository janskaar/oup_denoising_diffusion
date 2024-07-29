import os, time, h5py
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import orbax.checkpoint

from diffusion import norm_data
from unet import Encoder

from simulator import ParticleSimulator, SimulationParameters


top_dir = "results_reduced_parameterspace_norm_012_latent_4"
run_id = "47"

outfile = os.path.join(top_dir, f"line_encodings_run_{run_id}.h5")

ckpt_path = os.path.join(top_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
encoder_params = raw_restored["model"]["params"]["Encoder_0"]

X_path = os.path.join("data", "z_reduced_parameterspace.npy")
X = np.load(X_path)
Theta_path = os.path.join("data", "theta_reduced_parameterspace.npy")
Theta = np.load(Theta_path)

norm_data = partial(norm_data, Theta=Theta, axis=(0,1,2), base_X=X, base_Theta=Theta)

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
        sigma_noise=param_vals[0],
        tau_x=param_vals[1],
        tau_y=param_vals[2],
        C=param_vals[3],
    )

    simulator = ParticleSimulator(z0, params)
    simulator.simulate(3000)


    zs = simulator.z[:30000:10] # shape (time, num_procs, 2)

    x = zs.transpose(1,0,2) # shape (num_procs, time, 2)
    x, _ = norm_data(x)

    enc = Encoder(start_filters = 16,
                 filter_mults = (1, 2, 4, 8),
                 latent_dim = 4,
                 normalization = True,
                 activation = nn.silu).apply({"params": encoder_params}, x[:,1600:2624])
    return x, np.asarray(enc)

num_per_line = 21
line_ends = ( (1, 5), (5, 10), (1, 5), (5, 10) )
centers = np.tile(np.array([[3, 7.5, 3, 7.5]]), (num_per_line, 1))
params = ["sigma", "tau_x", "tau_y", "C"]

for i, ends in enumerate(line_ends):

    line = centers.copy()
    line[:,i] = np.linspace(ends[0], ends[1], num_per_line)
    xs = []
    encs = []
    for j, param in enumerate(line):
        print(f"Param {params[i]}, {j}/{num_per_line}", flush=True)
        x, enc = sim_and_encode(param, 500)
        xs.append(x)
        encs.append(enc)
    xs = np.array(xs)
    encs = np.array(encs)

    with h5py.File(outfile, "a") as f:
        grp = f.create_group(params[i])
        grp.create_dataset("x", data=xs)
        grp.create_dataset("encodings", data=encs)
        grp.create_dataset("line", data=line)


