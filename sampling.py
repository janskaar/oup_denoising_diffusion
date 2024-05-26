import jax.numpy as jnp
import jax 
from flax import jax_utils


def noise_to_x0(noise, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == noise.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None]
    x0 = 1. / sqrt_alpha_bar * xt -  jnp.sqrt(1./alpha_bar-1) * noise
    return x0


def x0_to_noise(x0, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == x0.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None]
    noise = (1. / sqrt_alpha_bar * xt - x0) /jnp.sqrt(1./alpha_bar-1)
    return noise


def get_posterior_mean_variance(img, t, x0, v, ddpm_params):

    beta = ddpm_params['betas'][t,None,None]
    alpha = ddpm_params['alphas'][t,None,None]
    alpha_bar = ddpm_params['alphas_bar'][t,None,None]
    alpha_bar_last = ddpm_params['alphas_bar'][t-1,None,None]
    sqrt_alpha_bar_last = ddpm_params['sqrt_alphas_bar'][t-1,None,None]

    # only needed when t > 0
    coef_x0 = beta * sqrt_alpha_bar_last / (1. - alpha_bar)
    coef_xt = (1. - alpha_bar_last) * jnp.sqrt(alpha) / ( 1- alpha_bar)        
    posterior_mean = coef_x0 * x0 + coef_xt * img

    posterior_variance = beta * (1 - alpha_bar_last) / (1. - alpha_bar)
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min = 1e-20))

    return posterior_mean, posterior_log_variance


def model_predict(state, x, t, condition, ddpm_params):
    variables = {'params': state.params}
    noise_pred = state.apply_fn(variables, x, t, condition)
    return noise_pred


def ddpm_sample_step(state, rng, x, t, condition, ddpm_params):
    print("Tracing ddpm_sample_step") 
    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t

    v = model_predict(state, x, batched_t, condition, ddpm_params)

    x_mean = 1 / jnp.sqrt(ddpm_params["alphas"][t]) * (x - ddpm_params["betas"][t] / ddpm_params["sqrt_1m_alphas_bar"][t] * v)
    var = (1 - ddpm_params["alphas_bar"][t-1]) / (1 - ddpm_params["alphas_bar"][t]) * ddpm_params["betas"][t]
    x = x_mean + jnp.sqrt(var) * jax.random.normal(rng, x.shape) 
    return x, x_mean

 
def sample_loop(rng, state, shape, condition, sample_step, timesteps):
    rng, x_rng = jax.random.split(rng)
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    print("Shape: ", shape)
    # sample step
    print("Sampling")
    for t in reversed(jnp.arange(timesteps)):
        print(t, end="\r")
        rng, step_rng = jax.random.split(rng)
        # x_mean is final prediction at t=0
        x, x_mean = sample_step(state, step_rng, x, t, condition)

    return x_mean

