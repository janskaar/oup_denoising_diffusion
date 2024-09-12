import jax
from flax import struct
import jax.numpy as jnp
import numpy as np
from functools import partial


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

@partial(jax.jit, static_argnums=-1)
@partial(jax.vmap, in_axes=(0, 0, None, None))
def mh_step(key, state, adapt, potential_fn):

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

@partial(jax.vmap, in_axes=(0, None))
def init_state(x, potential_fn, alpha=0.02, acc_target=0.7):
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


def find_last_accepted_warmups(states):
    num_chains = len(states[0].logp)
    s = states[0]
    mu = np.zeros_like(s.mu)
    cov = np.zeros_like(s.cov)
    cov_scale = np.zeros_like(s.cov_scale)
    x = np.zeros_like(s.x)
    logp = np.zeros_like(s.logp)
    accepted = np.zeros_like(s.accepted)
    acc_ma = np.zeros_like(s.acc_ma)
    alpha = np.zeros_like(s.alpha)
    acc_target = np.zeros_like(s.acc_target)
    for i in range(num_chains):
        states_ = [s for s in states if s.accepted[i]]
        state = states_[-1] 
        mu[i] = state.mu[i]
        cov[i] = state.cov[i]
        cov_scale[i] = state.cov_scale[i]
        x[i] = state.x[i]
        logp[i] = state.logp[i]
        alpha[i] = state.alpha[i]
        acc_target[i] = state.acc_target[i]

    state = MCMCState(mu = jnp.array(mu),
                      cov = jnp.array(cov),
                      cov_scale = jnp.array(cov_scale),
                      x = jnp.array(x),
                      logp = jnp.array(logp),
                      step = jnp.zeros(num_chains),
                      accepted = jnp.array(accepted),
                      acc_ma = jnp.array(acc_ma),
                      alpha = jnp.array(alpha),
                      acc_target = jnp.array(acc_target))

    return state





