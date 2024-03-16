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
import scanpy as sc

import sklearn.neighbors

class FeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    n_layers: int
    n_neurons: int
    n_output: int
    
    @nn.compact
    def __call__(self, x):
        
        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_output = self.n_output
        
        x = nn.Dense(
        features = n_neurons,
        dtype = jnp.float32,
        kernel_init = nn.initializers.glorot_uniform(),
        bias_init = nn.initializers.zeros_init()
        )(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm(dtype = jnp.float32)(x)
        
        for _ in range(n_layers-1):
            
            x = nn.Dense(
            features = n_neurons,
            dtype = jnp.float32,
            kernel_init = nn.initializers.glorot_uniform(),
            bias_init = nn.initializers.zeros_init()
            )(x)
            x = nn.leaky_relu(x) + x
            x = nn.LayerNorm(dtype = jnp.float32)(x)

        output = nn.Dense(
        features = n_output,
        dtype = jnp.float32,
        kernel_init = nn.initializers.glorot_uniform(),
        bias_init = nn.initializers.zeros_init()
        )(x)
        
        return output


    
class CVAE(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    n_layers: int
    n_neurons: int
    n_latent: int
    n_output_exp: int
    n_output_cov: int
    
    
    def setup(self):
        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_latent = self.n_latent
        n_output_exp = self.n_output_exp
        n_output_cov = self.n_output_cov
        
        self.encoder = FeedForward(n_layers = n_layers, 
                                   n_neurons = n_neurons, 
                                   n_output = n_latent*2)
                             
        self.decoder_exp = FeedForward(n_layers = n_layers, 
                                       n_neurons = n_neurons, 
                                       n_output = n_output_exp)
                             
        self.decoder_cov = FeedForward(n_layers = n_layers, 
                                   n_neurons = n_neurons, 
                                   n_output = n_output_cov)
        
    def __call__(self, x, mode = 'spatial', key = random.key(0)):
        
        
        conf_const = 0 if mode == 'spatial' else 1 
        conf_neurons = jax.nn.one_hot(conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32)
        
        x_conf = jnp.concatenate([x, conf_neurons], axis = -1)
        enc_mu, enc_logstd = jnp.split(self.encoder(x_conf), 2, axis = -1)
        
        key, subkey = random.split(key)
        z =  enc_mu + random.normal(key = subkey, shape = enc_logstd.shape) *  jnp.exp(enc_logstd)
        z_conf = jnp.concatenate([z, conf_neurons], axis = -1)
        
        dec_exp = self.decoder_exp(z_conf)
        
        if(mode == 'spatial'):
            dec_cov = self.decoder_cov(z_conf)
            return(enc_mu, enc_logstd, dec_exp, dec_cov)
        return(enc_mu, enc_logstd, dec_exp)


@struct.dataclass
class Metrics(metrics.Collection):
    enc_loss: metrics.Average.from_output('enc_loss')
    dec_loss: metrics.Average.from_output('dec_loss')
    enc_corr: metrics.Average.from_output('enc_corr')
    
class TrainState(train_state.TrainState):
    metrics: Metrics


def MatSqrt(Mats):
    """
    Computes psuedo matrix square root with tensorfow linear algebra on cpu

    Args:
        Mats (array): Matrices to compute square root of 
    Return:
        SqrtMats (np.array): psuedo matrix square of Mats
    """
    
    e,v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m,n = e.shape
    diag_e = np.zeros((m,n,n),dtype=e.dtype)
    diag_e.reshape(-1,n**2)[...,::n+1] = e
    return(np.matmul(np.matmul(v, diag_e), v.transpose([0,2,1])))


def BatchKNN(data, batch, k):
    
    kNNGraphIndex = np.zeros(shape = (data.shape[0], k))
    
    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]
        
        batch_knn = sklearn.neighbors.kneighbors_graph(data[val_ind], n_neighbors=k, mode='connectivity', n_jobs=-1).tocoo()
        batch_knn_ind = np.reshape(np.asarray(batch_knn.col), [data[val_ind].shape[0], k])
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]
        
    return(kNNGraphIndex.astype('int'))
    

def CalcCovMats(spatial_data, kNN, genes, spatial_key = 'spatial', batch_key = -1):
    
        
    """
    Wrapper to compute niche covariance based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbours to define niche
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        MeanExp (np.array): expression vector to shift niche covariance with
        weighted (bool): if True, weights covariance by spatial distance
    Return:
        CovMats: niche covariance matrices
        kNNGraphIndex: indices of nearest spatial neighbours per cell
    """
    ExpData = np.log(spatial_data[:, genes].X + 1)
    
    if(batch_key == -1):        
        kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=kNN, mode='connectivity', n_jobs=-1).tocoo()
        kNNGraphIndex = np.reshape(np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN])
    else:
        kNNGraphIndex = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)
    
        
    DistanceMatWeighted = (ExpData.mean(axis = 0)[None, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]])

    CovMats = np.matmul(DistanceMatWeighted.transpose([0,2,1]), DistanceMatWeighted) / (kNN - 1)
    CovMats = CovMats + CovMats.mean() * 0.00001 * np.expand_dims(np.identity(CovMats.shape[-1]), axis=0) 
    return(CovMats)

def compute_covet(spatial_data, k = 8, g = 64, genes = [], spatial_key = 'spatial', batch_key = -1):
    
    """
    Compte niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbours to define niche
        g (int): number of HVG to compute niche covariance matricies
        genes (list of str): list of genes to keep for niche covariance

    Return:
        CovMats: raw, untransformed niche covariance matrices
        CovMatsTransformed: covariance matrices transformed into chosen cov_dist
        NeighExp: Average geene expression in niche 
        CovGenes: Genes used for niche covariance 
    """
        
        

    if(g == -1):
        CovGenes = spatial_data.var_names
    else:
        if 'highly_variable' not in spatial_data.var.columns:
            spatial_data.layers['log'] = np.log(spatial_data.X+1)
            sc.pp.highly_variable_genes(spatial_data, n_top_genes = g, layer = 'log')
        
        
        CovGenes = np.asarray(spatial_data.var_names[spatial_data.var.highly_variable])
        if(len(genes) > 0):
            CovGenes = np.union1d(CovGenes, genes)
    
    CovMats = CalcCovMats(spatial_data, k, genes = CovGenes, spatial_key = spatial_key, batch_key = batch_key)
    CovMatsTransformed = MatSqrt(CovMats)

    
    return(CovMats.astype('float32'), CovMatsTransformed.astype('float32'), np.asarray(CovGenes).astype('str'))

