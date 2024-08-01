import os, time, h5py
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint
import ml_collections

from sampling import ddpm_sample_step
from diffusion import norm_data, get_ddpm_params
from default_config import config
from unet import UNET

from simulator import ParticleSimulator, SimulationParameters

top_dir = "results_reduced_parameterspace_norm_012_latent_4"
run_id = "47"
ckpt_path = os.path.join(top_dir, f"checkpoint_{run_id}")

def get_config(path):
    config = ml_collections.ConfigDict()
    with h5py.File(path, "r") as f:
        keys = list(f["config"].attrs.keys())
        keys_split = [k.split(".") for k in keys]        

        for k in keys_split:
            if len(k) == 1: # not part of any subdict
                setattr(config, k[0], f["config"].attrs[k[0]])
            else: # part of a subdict
                if not hasattr(config, k[0]):
                    # create subdict if it hasn't been created
                    setattr(config, k[0], ml_collections.ConfigDict())
                # set attribute to subdict 
                setattr(getattr(config, k[0]), k[1], f["config"].attrs[k[0] + "." + k[1]])
    return config

config = get_config(os.path.join(top_dir, f"run_{run_id}.h5"))

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
diffusion_params = raw_restored["model"]["params"]

train_X = np.load(os.path.join("data", "z_reduced_parameterspace.npy"))
train_Theta = np.load(os.path.join("data", "theta_reduced_parameterspace.npy"))

## Denoising model

model = UNET(
    start_filters=config.model.start_filters,
    filter_mults=config.model.filter_mults,
    out_channels=config.data.channels,
    activation=nn.silu,
    encoder_start_filters=config.model.encoder_start_filters,
    encoder_filter_mults=config.model.encoder_filter_mults,
    encoder_latent_dim=config.model.encoder_latent_dim,
    use_encoder=config.model.use_encoder,
    use_parameters=config.model.use_parameters,
    normalization=config.model.normalization
)


optimizer = optax.adam(1e-3)

state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=diffusion_params,
    tx=optimizer,
)

## Sampling functions

def sample_deterministic_and_return_every(rng, state, shape, condition, sample_step, timesteps, save_every=50):
    rng, x_rng = jax.random.split(rng)
    # generate the initial sample (pure noise)
    x_mean = jax.random.normal(x_rng, shape)
    # sample step
    print("Sampling")
    samples = [x_mean]
    for i, t in enumerate(reversed(jnp.arange(timesteps))):
        print(t, end="\r")
        rng, step_rng = jax.random.split(rng)
        # x_mean is final prediction at t=0
        x, x_mean = sample_step(state, step_rng, x_mean, t, condition)
        if (i + 1) % save_every == 0:
            samples.append(x_mean)
    return samples


def sample_and_return_every(rng, state, shape, condition, sample_step, timesteps, save_every=50):
    rng, x_rng = jax.random.split(rng)
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    # sample step
    print("Sampling")
    samples = [x]
    for i, t in enumerate(reversed(jnp.arange(timesteps))):
        print(t, end="\r")
        rng, step_rng = jax.random.split(rng)
        # x_mean is final prediction at t=0
        x, x_mean = sample_step(state, step_rng, x, t, condition)
        if (i + 1) % save_every == 0:
            samples.append(x)
    samples.pop(-1)
    samples.append(x_mean)
    return samples





## Main function

def simulate_and_sample_diffusion(sim_params, num_procs):
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
    z = z[::10, :].transpose(1,0,2) # reduce to 1 ms resolution, transpose to (n_proc, time, channel)
    z = z[:,-1024:] # use only last 1024 time bins

    # Condition for reverse process
    # tile, then reshape such that same simulation come consecutively
    x, _ = norm_data(z, train_Theta, base_X=train_X, axis=(0,1,2))


    ## Sample

    ddpm_params = get_ddpm_params(config.ddpm)

    sample_step = partial(ddpm_sample_step, ddpm_params=ddpm_params)
    sample_step = jax.jit(sample_step)

    rng = jax.random.PRNGKey(123)
    rng, key = jax.random.split(rng)

    sample = sample_and_return_every(
         key, state, x.shape, x, sample_step, config.ddpm.timesteps, save_every=50
    )
    sample = np.asarray(sample)
    deterministic_sample = sample_deterministic_and_return_every(
         key, state, x.shape, x, sample_step, config.ddpm.timesteps, save_every=50
    )
    deterministic_sample = np.asarray(deterministic_sample)
    return x, sample, deterministic_sample

## Run and save

sim_params = np.array([[4., 6., 2., 7.],
                       [1., 5., 5., 9.],
                       [5., 2., 8., 5.],
                       [3., 4., 1., 6.]])

num_procs = 100

for i, p in enumerate(sim_params):
    x, sample, deterministic_sample = simulate_and_sample_diffusion(p, num_procs)


    with h5py.File(os.path.join(top_dir, f"samples_reverse_process_run_{run_id}.h5"), "a") as f:
        grp = f.create_group(str(i))
        grp.create_dataset("samples", data=np.asarray(sample))
        grp.create_dataset("deterministic_samples", data=np.asarray(deterministic_sample))
        grp.create_dataset("conditions", data=x)
        grp.attrs["sigma_noise"] = p[0],
        grp.attrs["tau_x"] = p[1],
        grp.attrs["tau_y"] = p[2],
        grp.attrs["C"] = p[3],


