import os, time, h5py
import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import orbax.checkpoint

from diffusion import norm_data
from unet import Encoder

top_dir = "results_reduced_parameterspace_norm_012_latent_5"
run_id = "0"
ckpt_path = os.path.join(top_dir, f"checkpoint_{run_id}")

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
raw_restored = orbax_checkpointer.restore(ckpt_path)
params = raw_restored["model"]["params"]

encoder_params = raw_restored["model"]["params"]["Encoder_0"]
X_path = os.path.join("data", "z_reduced_parameterspace_test.npy")
X = np.load(X_path)
Theta_path = os.path.join("data", "theta_reduced_parameterspace_test.npy")
Theta = np.load(Theta_path)
X, Theta = norm_data(X, Theta, axis=(0,1))


def data_gen(batch_size, X, Theta):
    i = 0
    num = len(X) // batch_size
    if len(X) % batch_size != 0:
        num += 1

    while i <= num - 1:
        yield X[i * batch_size : (i + 1) * batch_size], Theta[
            i * batch_size : (i + 1) * batch_size
        ]
        i += 1


start_indices = jnp.arange(1024, 1624+1, 50)
encs = []
for start in start_indices:
    for (x, theta) in data_gen(500, X[:,start:start+1024], Theta):
        enc = Encoder(start_filters = 16,
                     filter_mults = (1, 2, 4, 8),
                     latent_dim = 5,
                     normalization = True,
                     activation = nn.silu).apply({"params": encoder_params}, x)
        encs.append(enc)  
encs = np.concatenate(encs)


np.save(os.path.join(top_dir, f"encodings_test_set_run_{run_id}.npy"), encs)

