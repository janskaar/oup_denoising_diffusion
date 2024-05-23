"""
Partly taken from denoising-diffusion-pytorch / denoising-diffusion-flax repositories
"""

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
    out = t/denom
    return (out)


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


class Attention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    scale: int = 10

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape                   # B is batch, N is length of time series, C num channels
        dim = self.dim_head * self.heads

        qkv = nn.Conv(features= dim * 3, kernel_size=(1,),
                      use_bias=False, name='to_qkv.conv_0')(x)  # [B, N, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, N, dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), (q, k, v)) # h = num heads,
                                                                                               # d = head dim
        assert q.shape == k.shape == v.shape == (
            B, N, self.heads, self.dim_head)

        q, k = map(l2norm, (q, k))

        sim = jnp.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale # (B, h, N, N)
        attn = nn.softmax(sim, axis=-1) # (B, h, N, N)
        assert attn.shape == (B, self.heads, N,  N)

        out = jnp.einsum('b h i j , b j h d  -> b h i d', attn, v) # (B, h, N, d)
        out = rearrange(out, 'b h x d -> b x (h d)') # (B, N, h * d)
        assert out.shape == (B, N, dim)

        out = nn.Conv(features=C, kernel_size=(1,), name='to_out.conv_0')(out) # (B, N, C)
        return out


class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape                   # B is batch, N is length of time series, C num channels
        dim = self.dim_head * self.heads



        qkv = nn.Conv(features= dim * 3, kernel_size=(1,),
                      use_bias=False, name='to_qkv.conv_0')(x)  # [B, N, dim *3]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # [B, N, dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), (q, k, v)) # h = num heads,
                                                                                               # d = head dim
 
        assert q.shape == k.shape == v.shape == (
            B, N, self.heads, self.dim_head)
        # compute softmax for q along its embedding dimensions
        q = nn.softmax(q, axis=-1)
        # compute softmax for k along its spatial dimensions
        k = nn.softmax(k, axis=-3)

        q = q/jnp.sqrt(self.dim_head)
        v = v / N

        context = jnp.einsum('b n h d, b n h e -> b h d e', k, v)
        out = jnp.einsum('b h d e, b n h d -> b h e n', context, q)
        out = rearrange(out, 'b h e n -> b n (h e)')
        assert out.shape == (B, N, dim)

        out = nn.Conv(features=C, kernel_size=(1,), name='to_out.conv_0')(out)
        out = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=jnp.float32, name='to_out.norm_0')(out)
        return out



class AttnBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True


    @nn.compact
    def __call__(self, x):
      B, W, C = x.shape
      normed_x = nn.LayerNorm(epsilon=1e-5, use_bias=False, dtype=jnp.float32)(x)
      if self.use_linear_attention:
        attn = LinearAttention(self.heads, self.dim_head)
      else:
        attn = Attention(self.heads, self.dim_head)
      out = attn(normed_x)
      assert out.shape == (B, W, C)
      return out + x

class ResidualBlock(nn.Module):

    filters : int
    activation : Callable

    @nn.compact
    def __call__(self, x, time_emb, param_emb):

        y = nn.Conv(self.filters, kernel_size = (3,), strides = 1, padding = "SAME")(x)
        y = nn.LayerNorm(reduction_axes = (-1, -2))(y)
        y = self.activation(y)

        # add in time and param embedding
        time_emb = nn.Dense(features=2 * self.filters,
                            name="time_embedding_mlp.dense_0")(nn.silu(time_emb)) # [B, 2T], T is time embedding dim
        time_emb = time_emb[:,  None, :]  # [B, 1, 2T]  # broadcast over signal dim

        param_emb = nn.Dense(features=2 * self.filters,
                            name="param_embedding_mlp.dense_0")(nn.silu(param_emb)) # [B, 2T], T is time embedding dim
        param_emb = param_emb[:,  None, :]  # [B, 1, 2T]  # broadcast over signal dim

        time_scale, time_shift = jnp.split(time_emb, 2, axis=-1)  # [B, 1, T]
        y = y * (1 + time_scale) + time_shift

        param_scale, param_shift = jnp.split(param_emb, 2, axis=-1)  # [B, 1, T]
        y = y * (1 + param_scale) + param_shift

        y = nn.Conv(self.filters, kernel_size = (3,), strides = 1, padding = "SAME")(y)
        y = nn.LayerNorm(reduction_axes = (-1, -2))(y)

        # restore channel dimension for ups
        if x.shape[-1] != self.filters:
            x = nn.Conv(self.filters, (1,))(x)

        return self.activation(y) + x

class Downsample(nn.Module):

    filters : int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, kernel_size = (3,), strides = 2, padding = "SAME")(x)
        return y

class Upsample(nn.Module):

    filters : int

    @nn.compact
    def __call__(self, x):
        y = nn.ConvTranspose(features=self.filters, strides=(2,), kernel_size=(3,))(x)
        return y


class UNET(nn.Module):

    start_filters : int
    filter_mults : Sequence[int]
    out_channels: int
    activation : Callable
    encoder_start_filters : int
    encoder_filter_mults : Sequence[int]
    encoder_latent_dim : int
    attention : bool

    use_encoder : bool

    @nn.compact
    def __call__(self, x, t, z):
        time_dim = self.start_filters * 4
        # use sinusoidal embeddings to encode timesteps
        time_emb = SinusoidalPosEmb(time_dim)(t)  # [B, dim]
        time_emb = nn.Dense(features=time_dim, name="time_mlp.dense_0")(time_emb) # [B, 4*dim]
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=time_dim,  name="time_mlp.dense_1")(time_emb)

        if self.use_encoder:
            param_emb = Encoder(
                    start_filters = self.encoder_start_filters,
                    filter_mults = self.encoder_filter_mults,
                    latent_dim = self.encoder_latent_dim,
                    activation = self.activation,
                    attention = self.attention)(z)
        else:
            param_emb = nn.Dense(features=time_dim, name="param_mlp.dense_0")(z)
            param_emb = nn.gelu(param_emb)
            param_emb = nn.Dense(features=time_dim,  name="param_mlp.dense_1")(param_emb)


        x = nn.Conv(
            features= self.start_filters * self.filter_mults[0] ,
            kernel_size=(3,), 
            padding="SAME",
            name="init.conv_0")(x)

        # down
        xs = []
        for i, mult in enumerate(self.filter_mults):
            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_0")(x, time_emb, param_emb)
            xs.append(x)

            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_1")(x, time_emb, param_emb)
            if self.attention:
                x = AttnBlock(name=f'down_{i}.attnblock_0')(x)
            xs.append(x)

            if i < len(self.filter_mults) - 1:
                x = Downsample(self.start_filters * self.filter_mults[i+1],
                               name=f"downsample_{i}")(x)

        # middle
        x = ResidualBlock(filters = self.start_filters * self.filter_mults[-1],
                          activation = self.activation,
                          name="middle_0")(x, time_emb, param_emb)
        if self.attention:
            x = AttnBlock(use_linear_attention=False, name = 'mid.attenblock_0')(x)
        x = ResidualBlock(filters = self.start_filters * self.filter_mults[-1],
                          activation = self.activation,
                          name="middle_1")(x, time_emb, param_emb)



        # up
        for i, mult in enumerate(reversed(self.filter_mults)):
            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"up_{i}_0")(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)

            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"up_{i}_1")(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)
            if self.attention:
                x = AttnBlock(name=f'up_{i}.attnblock_0')(x)

            if i < len(self.filter_mults) - 1:
                x = Upsample(self.start_filters * mult  // 2,
                             name=f"upsample_{i}")(x)

        x = ResidualBlock(filters = self.out_channels,
                          activation = self.activation,
                          name="final_resblock")(x, time_emb, param_emb)

        return x        


class UNETXENC(nn.Module):

    start_filters : int
    filter_mults : Sequence[int]
    out_channels: int
    activation : Callable

    @nn.compact
    def __call__(self, x, t, z):
        time_dim = self.start_filters * 4
        # use sinusoidal embeddings to encode timesteps
        time_emb = SinusoidalPosEmb(time_dim)(t)  # [B, dim]
        time_emb = nn.Dense(features=time_dim, name="time_mlp.dense_0")(time_emb) # [B, 4*dim]
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=time_dim,  name="time_mlp.dense_1")(time_emb)

        param_emb = z

        x = nn.Conv(
            features= self.start_filters * self.filter_mults[0],
            kernel_size=(3,), 
            padding="SAME",
            name="init.conv_0")(x)

        # down
        xs = []
        for i, mult in enumerate(self.filter_mults):
            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_0")(x, time_emb, param_emb)
            xs.append(x)

            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_1")(x, time_emb, param_emb)

            x = AttnBlock(name=f'down_{i}.attnblock_0')(x)
            xs.append(x)

            if i < len(self.filter_mults) - 1:
                x = Downsample(self.start_filters * self.filter_mults[i+1],
                               name=f"downsample_{i}")(x)

        # middle
        x = ResidualBlock(filters = self.start_filters * self.filter_mults[-1],
                          activation = self.activation,
                          name="middle_0")(x, time_emb, param_emb)
        x = AttnBlock(use_linear_attention=False, name = 'mid.attenblock_0')(x)
        x = ResidualBlock(filters = self.start_filters * self.filter_mults[-1],
                          activation = self.activation,
                          name="middle_1")(x, time_emb, param_emb)



        # up
        for i, mult in enumerate(reversed(self.filter_mults)):
            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"up_{i}_0")(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)

            x = ResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"up_{i}_1")(jnp.concatenate((xs.pop(), x), axis=-1), time_emb, param_emb)

            x = AttnBlock(name=f'up_{i}.attnblock_0')(x)

            if i < len(self.filter_mults) - 1:
                x = Upsample(self.start_filters * mult  // 2,
                             name=f"upsample_{i}")(x)

        x = ResidualBlock(filters = self.out_channels,
                          activation = self.activation,
                          name="final_resblock")(x, time_emb, param_emb)

        return x        





class EncodingResidualBlock(nn.Module):

    filters : int
    activation : Callable

    @nn.compact
    def __call__(self, x):

        y = nn.Conv(self.filters, kernel_size = (3,), strides = 1, padding = "SAME")(x)
        y = nn.LayerNorm(reduction_axes = (-1, -2))(y)
        y = self.activation(y)

        y = nn.Conv(self.filters, kernel_size = (3,), strides = 1, padding = "SAME")(y)
        y = nn.LayerNorm(reduction_axes = (-1, -2))(y)

        # restore channel dimension for ups
        if x.shape[-1] != self.filters:
            x = nn.Conv(self.filters, (1,))(x)

        return self.activation(y) + x




class Encoder(nn.Module):

    start_filters : int
    filter_mults : Sequence[int]
    latent_dim : int
    activation : Callable
    attention : bool

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features= self.start_filters * self.filter_mults[0] ,
            kernel_size=(3,), 
            padding="SAME",
            name="init.conv_0")(x)

        # down
        xs = []
        for i, mult in enumerate(self.filter_mults):
            x = EncodingResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_0")(x)
            xs.append(x)

            x = EncodingResidualBlock(filters = self.start_filters * mult,
                              activation = self.activation,
                              name=f"down_{i}_1")(x)

            if self.attention:
                x = AttnBlock(name=f'down_{i}.attnblock_0')(x)
            xs.append(x)

            if i < len(self.filter_mults) - 1:
                x = Downsample(self.start_filters * self.filter_mults[i+1],
                               name=f"downsample_{i}")(x)

        # middle
        x = EncodingResidualBlock(filters = self.start_filters * self.filter_mults[-1],
                          activation = self.activation,
                          name="middle_0")(x)
        if self.attention:
            x = AttnBlock(use_linear_attention=False, name = 'mid.attenblock_0')(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim)(x)
        return x        


if __name__ == "__main__": 
    import numpy as np
    import matplotlib.pyplot as plt

    dummy_X = np.random.normal(size=(5, 1024, 2))
    print(nn.tabulate(
        UNET(start_filters = 2,
             filter_mults = [1, 2, 4],
             activation = nn.relu),
        jax.random.key(0),
        depth=1)(dummy_X, np.array([5.])))


