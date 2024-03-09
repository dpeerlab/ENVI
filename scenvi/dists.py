import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state 
from clu import metrics

import jax
import jax.numpy as jnp
from jax import random


from functools import partial
import scipy.stats
import numpy as np

from typing import Callable, Any, Optional

import tensorflow_probability.substrates.jax.distributions as jnd

import scanpy as sc

def KL(mean, log_std):
    KL = 0.5 * (jnp.square(mean) + jnp.square(jnp.exp(log_std)) - 2 * log_std)
    return(jnp.mean(KL, axis = -1))
    
def log_pos_pdf(sample, l):
    log_prob = jnd.Poisson(rate=l).log_prob(sample)
    return(jnp.mean(log_prob, axis = -1))

def log_nb_pdf(sample, r, p):
    log_prob = jnd.NegativeBinomial(total_count=r, logits=p).log_prob(sample)
    return(jnp.mean(log_prob, axis = -1))

def log_zinb_pdf(sample, r, p, d):
    log_prob = jnd.Inflated(jnd.NegativeBinomial(total_count = r, logits = p), inflated_loc_logits = d).log_prob(sample)
    return(jnp.mean(log_prob, axis = -1))
    
def log_normal_pdf(sample, mean):
    log_prob = jnd.Normal(loc = mean, scale = 1).log_prob(sample)
    return(jnp.mean(log_prob, axis = -1))


def AOT_Distance(sample, mean):
    sample = jnp.reshape(sample, [sample.shape[0], -1])
    mean = jnp.reshape(mean, [mean.shape[0], -1])
    log_prob = - jnp.square(sample - mean)
    return(jnp.mean(log_prob, axis = -1))

