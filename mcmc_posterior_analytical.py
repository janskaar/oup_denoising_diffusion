from functools import partial
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from flax import struct
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
    logprob = tfd.MultivariateNormalTriL(loc=jnp.zeros(2048), scale_tril=jnp.linalg.cholesky(cov[:2048,:2048])).log_prob(x)
    return logprob


def compute_log_prob_normed_params(p, x):
    """
    Compute the log prob where the parameters have been rescaled to
    the unit cube
    """
    p = p * jnp.array([4, 5, 4, 5])
    p = p + jnp.array([1, 5, 1, 5])

    return compute_log_prob(p, x)



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

x = zs[0].T.reshape(-1)
potential_fn = partial(compute_log_prob_normed_params, x=x)

@struct.dataclass
class MCMCState:
    mu : jnp.array
    cov: jnp.array
    cov_scale : float
    x : jnp.array
    logp : jnp.array
    step : int
    accepted : bool
    acc_ma : float
    alpha : float      # for moving average of acceptance
    acc_target : float # target for tuning scale of proposal


def adapt_cov(state):
    def update_mu_covariance(state):
        mu = state.mu + (state.x - state.mu) / state.step
        residuals = state.x - state.mu
        cov = (
           state.cov * (state.step - 1) 
        + state.step / (state.step - 1) * jnp.outer(residuals, residuals)
        ) / state.step
        return mu, cov

    mu, cov = jax.lax.cond(state.accepted, update_mu_covariance, lambda state: (state.mu, state.cov), state)

    acc_ma = state.accepted * state.alpha + (1 - state.alpha) * state.acc_ma
    cov_scale = jax.lax.cond(acc_ma < state.acc_target, 
                       lambda cov_scale: cov_scale * 0.99,
                       lambda cov_scale: cov_scale * 1.01,
                       state.cov_scale)

    return MCMCState(
            mu = mu,
            cov = cov,
            cov_scale = cov_scale,
            x = state.x,
            logp = state.logp,
            step = state.step,
            accepted = state.accepted,
            acc_ma = acc_ma,
            alpha = state.alpha,
            acc_target = state.acc_target,
            )


def proposal(key, state):
    return jax.random.multivariate_normal(key, mean=jnp.zeros(4, dtype=np.float64), cov=state.cov * state.cov_scale)

@partial(jax.vmap, in_axes=(0, 0, None))
def mh_step(key, state, adapt):

    # Move
    rng, key = jax.random.split(key)
    x = state.x + proposal(key, state)

    # Check if any positions outside prior
    outside = (x > 1.).any() | (x < 0.).any()
    logp = potential_fn(x)

    # Compute ratio and correct for prior
    a = jnp.exp(logp - state.logp)
    a = jax.lax.cond(outside, lambda x: -1., lambda x: x, a)

    # Accept with probabilities ratio < u ~ U(0,1)
    rng, key = jax.random.split(rng)
    accept_prob = jax.random.uniform(rng)
    accepted = accept_prob < a
    state = jax.lax.cond(accepted, 
                         lambda state, x, logp: MCMCState(mu=state.mu,
                                                          cov=state.cov,
                                                          cov_scale=state.cov_scale,
                                                          x=x,
                                                          logp=logp,
                                                          step=state.step+1,
                                                          accepted=True,
                                                          acc_ma=state.acc_ma,
                                                          alpha=state.alpha,
                                                          acc_target=state.acc_target),
                         lambda state, x, logp: MCMCState(mu=state.mu,
                                                          cov=state.cov,
                                                          cov_scale=state.cov_scale,
                                                          x=state.x,
                                                          logp=state.logp,
                                                          step=state.step,
                                                          accepted=False,
                                                          acc_ma=state.acc_ma,
                                                          alpha=state.alpha,
                                                          acc_target=state.acc_target),
                         state,
                         x, 
                         logp)


    # adapt covariance matrix?
    state = jax.lax.cond(adapt, adapt_cov, lambda state: state, state)
    return state, x, logp

@partial(jax.vmap, in_axes=0)
def init_state(x, alpha=0.02, acc_target=0.7):
    logp = potential_fn(x)
    return MCMCState(
                     mu = jnp.zeros(4, dtype=np.float64),
                     cov = jnp.diag(jnp.ones(4, dtype=np.float64)),
                     cov_scale = jnp.array(0.001),
                     x = jnp.array(x),
                     logp = jnp.array(logp),
                     step = jnp.array(1),
                     accepted = jnp.array(True),
                     acc_ma = jnp.array(0.),
                     alpha = jnp.array(alpha),
                     acc_target = jnp.array(acc_target),
                     ) 

num_chains = 20
state = init_state(jnp.tile(jnp.array([[0.5, 0.5, 0.5, 0.5]]), (num_chains, 1)))
rng = jax.random.PRNGKey(123)

warmup_states = [state]
for i in range(1000):
    print(i, end="\r")
    rng, key = jax.random.split(rng)
    key = jax.random.split(key, num_chains)
    state, x, proposal_logp = mh_step(key, warmup_states[-1], True)
    warmup_states.append(state)

states = [warmup_states[-1]]
prop_logps = []
xs = []
for i in range(5000):
    print(i, end="\r")
    rng, key = jax.random.split(rng)
    state, x, proposal_logp = mh_step(key, states[-1], False)
    states.append(state)
    prop_logps.append(proposal_logp)
    xs.append(x)

sample = np.array([s.x for s in states if s.accepted])




##

# fig, ax = plt.subplots(4, 4)
# phi = np.arctan(1080 / 1920)
# sz = (14 * np.cos(phi), 14 * np.sin(phi))
# fig.set_size_inches(*sz)
# fig.subplots_adjust(wspace=0.4, hspace=0.4)
# 
# param_names = ["Sigma", "tau_x", "tau_y", "C"]
# encoding_names = [f"Encoding_{i}" for i in range(6)]
# 
# for i in range(4):
#     for j in range(4):
#         ax[i,j].scatter(sample[:,j], sample[:,i], s=2)
# 
# plt.show()
# 

