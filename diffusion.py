import jax
import jax.numpy as jnp
import numpy as np


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(beta_start, beta_end, timesteps, dtype=jnp.float32)
    return betas


def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps

    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar = jnp.sqrt(1.0 - alphas_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "sqrt_alphas_bar": sqrt_alphas_bar,
        "sqrt_1m_alphas_bar": sqrt_1m_alphas_bar,
    }


def l2_loss(logit, target):
    return (logit - target) ** 2


def simple_loss(params, state, inp, noise):
    x_t, condition, batched_t = inp
    pred = state.apply_fn({"params": params}, x_t, batched_t, condition)

    loss = l2_loss(
        pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
    )

    return loss.mean()


def full_loss(params, state, inp, noise, ddpm_params):
    x_t, condition, batched_t = inp
    pred = state.apply_fn({"params": params}, x_t, batched_t, condition)

    loss = l2_loss(
        pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
    )

    loss = loss / (
        2
        * ddpm_params["alphas"][batched_t, None]
        * (1 - ddpm_params["alphas_bar"][batched_t, None])
    )

    return loss.mean()


def compute_loss_full_chain(rng, state, X, condition, ddpm_params):
    """
    Compute loss over all diffusion time steps.
    Jit this function in order to avoid re-compiling inner loss function.
    """

    t = jnp.tile(jnp.arange(1000)[:, None], (1, len(X)))

    def loss_at_t(carry, t):
        """
        Carries only rng key, all other variables are static
        """
        print("Tracing loss_at_t")

        rng, key = jax.random.split(carry["key"])
        noise = jax.random.normal(carry["key"], X.shape)
        x_t = q_sample(X, t, noise, ddpm_params)
        pred = state.apply_fn({"params": state.params}, x_t, t, condition)

        loss = l2_loss(
            pred.reshape((pred.shape[0], -1)), noise.reshape((noise.shape[0], -1))
        )

        carry = {"key": rng}
        return carry, loss.mean(-1)

    carry = {"key": rng}
    carry, losses = jax.lax.scan(loss_at_t, carry, xs=t)
    return losses


def q_sample(x, t, noise, ddpm_params):
    sqrt_alpha_bar = ddpm_params["sqrt_alphas_bar"][t, None, None]
    sqrt_1m_alpha_bar = ddpm_params["sqrt_1m_alphas_bar"][t, None, None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t


