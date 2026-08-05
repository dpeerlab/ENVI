"""Flax/CLU model components for the ENVI CVAE.

Kept apart from ``utils.py`` so that COVET — which is pure numpy/sklearn/scanpy
— can be imported without jax, flax, clu or tensorflow_probability. Importing
this module requires the ``envi`` extra.
"""

import jax
import jax.numpy as jnp
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import random

class FeedForward(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_output: int

    @nn.compact
    def __call__(self, x):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_output = self.n_output

        x = nn.Dense(
            features=n_neurons,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x)

        for _ in range(n_layers - 1):

            x = nn.Dense(
                features=n_neurons,
                dtype=jnp.float32,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = nn.leaky_relu(x) + x
            x = nn.LayerNorm(dtype=jnp.float32)(x)

        output = nn.Dense(
            features=n_output,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)

        return output


class CVAE(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_latent: int
    n_output_exp: int
    n_output_cov: int

    def setup(self):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_latent = self.n_latent
        n_output_exp = self.n_output_exp
        n_output_cov = self.n_output_cov

        self.encoder = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_latent * 2
        )

        self.decoder_exp = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_exp
        )

        self.decoder_cov = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_cov
        )

    def __call__(self, x, mode="spatial", key=random.key(0)):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)
        enc_mu, enc_logstd = jnp.split(self.encoder(x_conf), 2, axis=-1)

        key, subkey = random.split(key)
        z = enc_mu + random.normal(key=subkey, shape=enc_logstd.shape) * jnp.exp(
            enc_logstd
        )
        z_conf = jnp.concatenate([z, conf_neurons], axis=-1)

        dec_exp = self.decoder_exp(z_conf)

        if mode == "spatial":
            dec_cov = self.decoder_cov(z)
            return (enc_mu, enc_logstd, dec_exp, dec_cov)
        return (enc_mu, enc_logstd, dec_exp)


@struct.dataclass
class Metrics(metrics.Collection):
    """
    :meta private:
    """

    enc_loss: metrics.Average
    dec_loss: metrics.Average
    enc_corr: metrics.Average


class TrainState(train_state.TrainState):
    """
    :meta private:
    """

    metrics: Metrics
