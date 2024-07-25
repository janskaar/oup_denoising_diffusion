from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jexpm
import jax.scipy as jsp
import einops
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from numpyro.infer import MCMC, NUTS


from simulator import SimulationParameters, ParticleSimulator



jax.config.update("jax_enable_x64", True)

# ##  Sanity check
num_procs = 10000
params = SimulationParameters(
    num_procs=num_procs,
    sigma_noise=4.0,
    tau_x=7,
    tau_y=2,
    C=9,
)

z0 = np.zeros((num_procs, 2))

simulator = ParticleSimulator(z0, params)
simulator.simulate(3000)
zs = simulator.z[:30000:10][-1024:].transpose((1,0,2))

# delta_s = jnp.arange(1024)
# cov = compute_ou_temporal_covariance(delta_s, params)
# cov_sim = compute_sample_temporal_covariance(zs[:,0], zs)
# 
# 
# 
# fig, ax = plt.subplots(2, 2, sharex=True)
# fig.set_size_inches(10,10)
# 
# ax[0,0].plot(cov_sim[:,0,0], label="simulation")
# ax[0,0].plot(cov[0,:1024], label="theory")
# ax[0,0].legend()
# 
# ax[0,1].plot(cov_sim[:,0,1])
# ax[0,1].plot(cov[0,1024:])
# 
# ax[1,0].plot(cov_sim[:,1,0])
# ax[1,0].plot(cov[1024,:1024])
# 
# ax[1,1].plot(cov_sim[:,1,1])
# ax[1,1].plot(cov[1024,1024:])
# 
# ax[0,0].set_xlim(0, 50)
# 
# plt.show()
# 
# ## 


def compute_stationary_covariance(p):
    """
    Computes the stationary covariance of the O-U process with parameters.
    The naming from the notes are as follows:
        a_1 = 1 / p[2]
        a_2 = -1 / p[3]
        a_3 = 1 / p[1]
        b_0 = p[0]
    """
    A = jnp.array([[1 / p[2], 0               ],
                  [-1 / p[3],     1 / p[1]]])
    s1 = p[0]**2 / (2 * A[0,0]) 
    s2 = -( A[1,0] * p[0]**2 ) / ( 2 * A[0,0] * (A[0,0] + A[1,1]) )
    s3 = ( A[1,0]**2 * p[0]**2 ) / ( 2 * A[0,0] * A[1,1] * (A[0,0] + A[1,1]) )
    return  jnp.array([[s1, s2],
                       [s2, s3]])


def compute_ou_temporal_covariance(p):
    """
    Computes the full covariance of the O-U process with 1024 time steps
    Computes the stationary covariance of the O-U process with parameters.
    The naming from the notes are as follows:
        a_1 = 1 / p[2]
        a_2 = -1 / p[3]
        a_3 = 1 / p[1]
        b_0 = p[0]
    """

    cov_s = compute_stationary_covariance(p)
    delta_s = jnp.arange(1024)
    A = jnp.array([[1 / p[2], 0       ],
                  [-1 / p[3], 1 / p[1]]])

    @partial(jax.vmap, in_axes=(None, 0))
    def vectorized_expm(A, s):
        return jexpm(- A * s)

    expAdelta = vectorized_expm(A, delta_s)

    # matmul along last two dimensions, broadcast along first (time) dimension
    cov_row = einops.einsum(cov_s, expAdelta.transpose(0,2,1), "j k,i k l->i j l")

    # make a new, symmetric vector that we can slice for the rows in the covariance matrix
    cov_row_sym = jnp.concatenate((jnp.flip(cov_row[1:].transpose(0,2,1), axis=0), cov_row))


    @partial(jax.vmap, in_axes=(1, None))
    @partial(jax.vmap, in_axes=(1, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def create_block_row(row, i):
        return jax.lax.dynamic_slice_in_dim(row, i, 1024)

    indices = jnp.arange(1024)[::-1]

    blocks = create_block_row(cov_row_sym, indices)

    cov = einops.rearrange(blocks, "c1 c2 s1 s2 -> (c1 s1) (c2 s2)")
    return cov


def compute_log_prob(p, x):
    """
    Computes the log probability of `x` under the distribution of the OU-process
    with parameters `p`.
    """

    cov = compute_ou_temporal_covariance(p)
    logprob = tfd.MultivariateNormalTriL(loc=jnp.zeros(128), scale_tril=jnp.linalg.cholesky(cov[:128,:128])).log_prob(x)
    return logprob

x = zs[0].T.reshape(-1)
params2 = SimulationParameters(
    num_procs=num_procs,
    sigma_noise=4.0,
    tau_x=7,
    tau_y=5,
    C=9,
)

p = jnp.array([params.sigma_noise,
               params.tau_x,
               params.tau_y,
               params.C])


pot = partial(compute_log_prob, x=x[:128])
## 
p2 = jnp.array([params.sigma_noise,
               params.tau_x * 10,
               params.tau_y * 1000,
               params.C])

pot(p2)
##
logprob1 = compute_log_prob(p, x[:128])

kernel = NUTS(potential_fn=partial(compute_log_prob, x=x[:128]))
num_samples = 20
mcmc = MCMC(kernel, num_warmup=10, num_samples=num_samples)
rng = jax.random.PRNGKey(123)
rng, key = jax.random.split(rng)
mcmc.run(key, init_params=p)



