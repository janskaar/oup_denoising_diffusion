import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from einops import rearrange
import numpy as np
from typing import Callable, Sequence
import os
import matplotlib.pyplot as plt


def l2norm(t, axis=1, eps=1e-12):
    """Performs L2 normalization of inputs over specified axis.

    Args:
      t: jnp.ndarray of any shape
      axis: the dimension to reduce, default -1
      eps: small value to avoid division by zero. Default 1e-12
    Returns:
      normalized array of same shape as t


    """
    denom = jnp.clip(jnp.linalg.norm(t, ord=2, axis=axis, keepdims=True), eps)
    out = t / denom
    return out


class SinusoidalPosEmb(nn.Module):

    dim: int

    @nn.compact
    def __call__(self, time):

        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class ResidualBlock(nn.Module):

    filters: int
    activation: Callable
    param_embedding : bool
    normalization : bool

    @nn.compact
    def __call__(self, x, time_emb, param_emb):

        y = nn.Conv(self.filters, kernel_size=(3,), strides=1, padding="SAME")(x)
        if self.normalization:
            y = nn.LayerNorm(reduction_axes=(-1, -2))(y)
        y = self.activation(y)

        # add in time and param embedding
        time_emb = nn.Dense(
            features=2 * self.filters, name="time_embedding_mlp.dense_0"
        )(
            nn.silu(time_emb)
        )  # [B, 2T], T is time embedding dim
        time_emb = time_emb[:, None, :]  # [B, 1, 2T]  # broadcast over signal dim

        time_scale, time_shift = jnp.split(time_emb, 2, axis=-1)  # [B, 1, T]
        y = y * (1 + time_scale) + time_shift

        if self.param_embedding:
            param_emb = nn.Dense(
                features=2 * self.filters, name="param_embedding_mlp.dense_0"
            )(
                nn.silu(param_emb)
            )  # [B, 2T], T is time embedding dim
            param_emb = param_emb[:, None, :]  # [B, 1, 2T]  # broadcast over signal dim


            param_scale, param_shift = jnp.split(param_emb, 2, axis=-1)  # [B, 1, T]
            y = y * (1 + param_scale) + param_shift

        y = nn.Conv(self.filters, kernel_size=(3,), strides=1, padding="SAME")(y)
        if self.normalization:
            y = nn.LayerNorm(reduction_axes=(-1, -2))(y)

        # restore channel dimension for ups
        if x.shape[-1] != self.filters:
            x = nn.Conv(self.filters, (1,))(x)

        return self.activation(y) + x


class Downsample(nn.Module):

    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, kernel_size=(3,), strides=2, padding="SAME")(x)
        return y


class Upsample(nn.Module):

    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.ConvTranspose(features=self.filters, strides=(2,), kernel_size=(3,))(x)
        return y


class UNET(nn.Module):

    start_filters: int
    filter_mults: Sequence[int]
    out_channels: int
    activation: Callable
    encoder_start_filters: int
    encoder_filter_mults: Sequence[int]
    encoder_latent_dim: int
    normalization : bool

    use_encoder: bool
    use_parameters : bool

    @nn.compact
    def __call__(self, x, t, z):
        time_dim = self.start_filters * 4
        # use sinusoidal embeddings to encode timesteps
        time_emb = SinusoidalPosEmb(time_dim)(t)  # [B, dim]
        time_emb = nn.Dense(features=time_dim, name="time_mlp.dense_0")(
            time_emb
        )  # [B, 4*dim]
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=time_dim, name="time_mlp.dense_1")(time_emb)

        if self.use_encoder:
            param_emb = Encoder(
                start_filters=self.encoder_start_filters,
                filter_mults=self.encoder_filter_mults,
                latent_dim=self.encoder_latent_dim,
                activation=self.activation,
            )(z)
        elif self.use_parameters:
            param_emb = nn.Dense(features=time_dim, name="param_mlp.dense_0")(z)
            param_emb = nn.gelu(param_emb)
            param_emb = nn.Dense(features=time_dim, name="param_mlp.dense_1")(param_emb)

        z_conditioning = self.use_encoder or self.use_parameters

        x = nn.Conv(
            features=self.start_filters * self.filter_mults[0],
            kernel_size=(3,),
            padding="SAME",
            name="init.conv_0",
        )(x)

        # down
        xs = []
        for i, mult in enumerate(self.filter_mults):
            x = ResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                param_embedding=z_conditioning,
                normalization=self.normalization,
                name=f"down_{i}_0",
            )(x, time_emb, param_emb)
            xs.append(x)

            x = ResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                param_embedding=z_conditioning,
                normalization=self.normalization,
                name=f"down_{i}_1",
            )(x, time_emb, param_emb)
            xs.append(x)

            if i < len(self.filter_mults) - 1:
                x = Downsample(
                    self.start_filters * self.filter_mults[i + 1],
                    name=f"downsample_{i}",
                )(x)

        # middle
        x = ResidualBlock(
            filters=self.start_filters * self.filter_mults[-1],
            activation=self.activation,
            param_embedding=z_conditioning,
            normalization=self.normalization,
            name="middle_0",
        )(x, time_emb, param_emb)
        x = ResidualBlock(
            filters=self.start_filters * self.filter_mults[-1],
            activation=self.activation,
            param_embedding=z_conditioning,
            normalization=self.normalization,
            name="middle_1",
        )(x, time_emb, param_emb)

        # up
        for i, mult in enumerate(reversed(self.filter_mults)):
            x = ResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                param_embedding=z_conditioning,
                normalization=self.normalization,
                name=f"up_{i}_0",
            )(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)

            x = ResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                param_embedding=z_conditioning,
                normalization=self.normalization,
                name=f"up_{i}_1",
            )(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)

            if i < len(self.filter_mults) - 1:
                x = Upsample(self.start_filters * mult // 2, name=f"upsample_{i}")(x)

        x = ResidualBlock(
            filters=self.out_channels,
            activation=self.activation, 
            param_embedding=z_conditioning,
            normalization=self.normalization,
            name="final_resblock"
        )(x, time_emb, param_emb)

        return x


class EncodingResidualBlock(nn.Module):

    filters: int
    activation: Callable
    normalization : bool

    @nn.compact
    def __call__(self, x):

        y = nn.Conv(self.filters, kernel_size=(3,), strides=1, padding="SAME")(x)
        if self.normalization:
            y = nn.LayerNorm(reduction_axes=(-1, -2))(y)
        y = self.activation(y)

        y = nn.Conv(self.filters, kernel_size=(3,), strides=1, padding="SAME")(y)
        if self.normalization:
            y = nn.LayerNorm(reduction_axes=(-1, -2))(y)

        # restore channel dimension for ups
        if x.shape[-1] != self.filters:
            x = nn.Conv(self.filters, (1,))(x)

        return self.activation(y) + x


class Encoder(nn.Module):

    start_filters: int
    filter_mults: Sequence[int]
    latent_dim: int
    activation: Callable

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features=self.start_filters * self.filter_mults[0],
            kernel_size=(3,),
            padding="SAME",
            name="init.conv_0",
        )(x)

        # down
        xs = []
        for i, mult in enumerate(self.filter_mults):
            x = EncodingResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                normalization=self.normalization,
                name=f"down_{i}_0",
            )(x)
            xs.append(x)

            x = EncodingResidualBlock(
                filters=self.start_filters * mult,
                activation=self.activation,
                normalization=self.normalization,
                name=f"down_{i}_1",
            )(x)

            xs.append(x)

            if i < len(self.filter_mults) - 1:
                x = Downsample(
                    self.start_filters * self.filter_mults[i + 1],
                    name=f"downsample_{i}",
                )(x)

        # middle
        x = EncodingResidualBlock(
            filters=self.start_filters * self.filter_mults[-1],
            activation=self.activation,
            normalization=self.normalization,
            name="middle_0",
        )(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    dummy_X = np.random.normal(size=(5, 1024, 2))
    print(
        nn.tabulate(
            UNET(start_filters=2, filter_mults=[1, 2, 4], activation=nn.relu),
            jax.random.key(0),
            depth=1,
        )(dummy_X, np.array([5.0]))
    )
