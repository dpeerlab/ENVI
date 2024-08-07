import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import random
from math import sqrt


# def MaxMinScale(arr):
#     """
#     :meta private:
#     """

#     arr = (
#         2
#         * (arr - arr.min(axis=0, keepdims=True))
#         / (arr.max(axis=0, keepdims=True) - arr.min(axis=0, keepdims=True))
#         - 1
#     )
#     return arr


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
    # n_output_cov: int
    k_nearest: int
    n_niche_genes: int

    def setup(self):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_latent = self.n_latent
        n_output_exp = self.n_output_exp
        # n_output_cov = self.n_output_cov
        k_nearest = self.k_nearest
        n_niche_genes = self.n_niche_genes

        self.encoder = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_latent * 2
        )

        self.decoder_exp = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_exp
        )

        # self.decoder_cov = FeedForward(
        #     n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_cov 
        # ) 
        # incorporated this into AttentionDecoderModel

        self.decoder_niche = AttentionDecoderModel(
            n_layers=n_layers,
            n_neurons=n_neurons,
            config=DefaultConfig(),
            out_seq_len=k_nearest,
            inp_dim=n_niche_genes
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
            # dec_cov = self.decoder_cov(z_conf)
            # return (enc_mu, enc_logstd, dec_exp, dec_cov)
            dec_niche = self.decoder_niche(z_conf)
            return (enc_mu, enc_logstd, dec_exp, dec_niche)
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


def MatSqrt(Mats):
    """
    :meta private:
    """

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def BatchKNN(data, batch, k):
    """
    :meta private:
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def CalcNicheMats(spatial_data, kNN, spatial_key="spatial", batch_key=-1):
    """
    :meta private:
    """

    ExpData = spatial_data.layers["scaled_log"] # constructed using scaled log data

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    NicheMats = (
        ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]
    )

    return NicheMats


def CalcCovMats(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1):
    """
    :meta private:
    """

    ExpData = np.log(spatial_data[:, genes].X + 1)

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    DistanceMatWeighted = (
        ExpData.mean(axis=0)[None, None, :]
        - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]
    )

    CovMats = np.matmul(
        DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted
    ) / (kNN - 1)
    CovMats = CovMats + CovMats.mean() * 0.00001 * np.expand_dims(
        np.identity(CovMats.shape[-1]), axis=0
    )
    return CovMats


def niche_cell_type(
    spatial_data, kNN, spatial_key="spatial", cell_type_key="cell_type", batch_key=-1
):
    """
    :meta private:
    """

    from sklearn.preprocessing import OneHotEncoder

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        knn_index = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        knn_index = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    one_hot_enc = OneHotEncoder().fit(
        np.asarray(list(set(spatial_data.obs[cell_type_key]))).reshape([-1, 1])
    )
    cell_type_one_hot = (
        one_hot_enc.transform(
            np.asarray(spatial_data.obs[cell_type_key]).reshape([-1, 1])
        )
        .reshape([spatial_data.obs["cell_type"].shape[0], -1])
        .todense()
    )

    cell_type_niche = pd.DataFrame(
        cell_type_one_hot[knn_index].sum(axis=1),
        index=spatial_data.obs_names,
        columns=list(one_hot_enc.categories_[0]),
    )
    return cell_type_niche

def compute_covet(
    spatial_data, k=8, g=64, genes=[], spatial_key="spatial", batch_key=-1
):
    """
    Compute niche covariance matrices for spatial data, run with scenvi.compute_covet

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param g: (int) number of HVG to compute COVET representation on (default 64)
    :param genes: (list of str) list of genes to keep for niche covariance (default []
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)

    :return COVET: niche covariance matrices
    :return COVET_SQRT: matrix square-root of niche covariance matrices for approximate OT
    :return CovGenes: list of genes selected for COVET representation
    """

    if g == -1:
        CovGenes = spatial_data.var_names
    else:
        if "highly_variable" not in spatial_data.var.columns:
            spatial_data.layers["log"] = np.log(spatial_data.X + 1)
            sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")

        CovGenes = np.asarray(spatial_data.var_names[spatial_data.var.highly_variable])
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)

    if batch_key not in spatial_data.obs.columns:
        batch_key = -1

    COVET = CalcCovMats(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, batch_key=batch_key
    )
    COVET_SQRT = MatSqrt(COVET)

    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes).astype("str"),
    )

def compute_niche(
        spatial_data, n_niche_genes, k=8, spatial_key="spatial", batch_key=-1
):
    """
    Compute niche matrices for spatial data in log space

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)

    :return Niches: niche matrices
    :return CovGenes: list of genes selected for COVET representation
    """

    # get min max of each gene in spatial_data and scale to -1 to 1
    gene_mins, gene_maxs = spatial_data.layers['log'].min(axis=0), spatial_data.layers['log'].max(axis=0)
    spatial_data.layers["scaled_log"] = ((spatial_data.layers["log"] - gene_mins) / (gene_maxs - gene_mins) * 2 - 1) / sqrt(n_niche_genes) #divide by n_niche_genes

    if batch_key not in spatial_data.obs.columns:
        batch_key = -1

    Niches = CalcNicheMats(
        spatial_data, k, spatial_key=spatial_key, batch_key=batch_key
    )

    spatial_data.obsm["scaled_niche"] = Niches.astype("float32")
    spatial_data.obsm["niche"] = (spatial_data.obsm["scaled_niche"] * sqrt(n_niche_genes) + 1) / 2 * (gene_maxs - gene_mins) + gene_mins # still in log space


    return (
        gene_mins, 
        gene_maxs
    )






# from Wasserstein Wormhole
from typing import Callable, Any, Optional
from jax.typing import ArrayLike


@struct.dataclass
class DefaultConfig:
    
    """
    Object with configuration parameters for Wormhole
    
    
    :param dtype: (data type) float point precision for Wormhole model (default jnp.float32)
    :param dist_func_enc: (str) OT metric used for embedding space (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param dist_func_dec: (str) OT metric used for Wormhole decoder loss (default 'S2', could be 'W1', 'S1', 'W2', 'S2', 'GW' and 'GS') 
    :param eps_enc: (float) entropic regularization for embedding OT (default 0.1)
    :param eps_dec: (float) entropic regularization for Wormhole decoder loss (default 0.1)
    :param lse_enc: (bool) whether to use log-sum-exp mode or kernel mode for embedding OT (default False)
    :param lse_dec: (bool) whether to use log-sum-exp mode or kernel mode for decoder OT (default True)
    :param coeff_dec: (float) coefficient for decoder loss (default 1)
    :param scale: (str) how to scale input point clouds ('min_max_total' and scales all point clouds so values are between -1 and 1)
    :param factor: (float) multiplicative factor applied on point cloud coordinates after scaling (default 1)
    :param emb_dim: (int) Wormhole embedding dimention (defulat 128)
    :param num_heads: (int) number of heads in multi-head attention (default 4)
    :param num_layers: (int) number of layers of multi-head attention for Wormhole encoder and decoder (default 3)
    :param qkv_dim: (int) dimention of query, key and value attributes in attention (default 512)
    :param mlp_dim: (int) dimention of hidden layer for fully-connected network after every multi-head attention layer
    :param attention_dropout_rate: (float) dropout rate for attention matrices during training (default 0.1)
    :param kernel_init: (Callable) initializer of kernel weights (default nn.initializers.glorot_uniform())
    :param bias_init: ((Callable) initializer of bias weights (default nn.initializers.zeros_init())
    """ 
    
    dtype: Any = jnp.float32
    dist_func_enc: str = 'S2'
    dist_func_dec: str = 'S2'
    eps_enc: float = 0.1
    eps_dec: float = 0.01
    lse_enc: bool = False
    lse_dec: bool = True
    coeff_dec: float = 1
    scale: str = 'min_max_total'
    factor: float = 1.0
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    qkv_dim: int = 512
    mlp_dim: int = 512
    attention_dropout_rate: float = 0.1
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros_init()

def expand_weights(weights):
    if weights.ndim == 2:
        weights = weights[:, None, None, :]
    if weights.ndim == 3:
        weights = weights.unsqueeze(1)
    while weights.ndim < 4:
        weights = weights.unsqueeze(0)
    return weights

def scaled_dot_product(q,
                       k,
                       v,
                       weights: Optional[ArrayLike] = None, 
                       scale_weights: float = 1,
                       deterministic: bool = False,
                       dropout_rng: Optional[ArrayLike] = random.key(0),
                       dropout_rate: float = 0.0,
                       ):
    
    dtype, d_k = q.dtype, q.shape[-1], 
    
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    

    if weights is not None:
        # attn_logits = attn_logits + jnp.tan(math.pi*(jnp.clip(weights, 1e-7, 1-1e-7)-1/2)) - jnp.tan(math.pi*(1/q.shape[-2]-1/2)) 
        attn_logits = attn_logits + jnp.log(weights/scale_weights + jnp.finfo(jnp.float32).tiny) 
        attn_logits = jnp.where(weights == 0, -9e15, attn_logits)
        attn_logits = jnp.where(weights == 1, 9e15, attn_logits)
        
    attention = nn.softmax(attn_logits, axis=-1)
    
      # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep = random.bernoulli(dropout_rng, keep_prob, attention.shape)  # type: ignore
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attention = attention * multiplier

    values = jnp.matmul(attention, v)
    return values, attention

class WormholeFeedForward(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.relu(x)
        output = nn.Dense(
            inputs.shape[-1],
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x) + inputs
        return output

class WeightedMultiheadAttention(nn.Module):
    
    config: DefaultConfig
    scale_weights: Optional[float] = 1
    
    def setup(self):
        config = self.config
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3 * config.emb_dim,
                                 dtype=config.dtype,
                                 kernel_init=config.kernel_init,
                                 bias_init=config.bias_init)
        
    def __call__(self, 
                 x,
                 weights: Optional[ArrayLike] = None, 
                 deterministic: Optional[bool] = True, 
                 dropout_rng: Optional[ArrayLike] = random.key(0)):
        
        config = self.config
        scale_weights = self.scale_weights
        
        batch_size, seq_length, _ = x.shape
        
        # assert x.shape[-1] == config.emb_dim
            
        if weights is not None:
            weights = expand_weights(weights)
            
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, config.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, weights = weights,
                                                        scale_weights = scale_weights,
                                                        deterministic = deterministic,
                                                        dropout_rng = dropout_rng,
                                                        dropout_rate = config.attention_dropout_rate)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, config.emb_dim)

        return values
    

class DecoderBlock(nn.Module):
    """Transformer decoder layer.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """

    config: DefaultConfig

    @nn.compact
    def __call__(self, inputs, deterministic, dropout_rng):
        config = self.config

        # Attention block.
        x = WeightedMultiheadAttention(config)(x = inputs, 
                                               deterministic = deterministic, 
                                               dropout_rng = dropout_rng) + inputs

        #x = nn.Dropout(rate=config.attention_dropout_rate)(x, deterministic=deterministic)
        x = nn.LayerNorm(dtype=config.dtype)(x)
        x = WormholeFeedForward(config=config)(x)
        output = nn.LayerNorm(dtype=config.dtype)(x)
        return output


class Unembedding(nn.Module):
    """Transformer embedding block.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
    """

    config: DefaultConfig
    inp_dim: int

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        output = nn.Dense(
            self.inp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        return output
    

class AttentionDecoderModel(nn.Module):
    """Transformer decoder network.

    Attributes:
    config: DefaultConfig dataclass containing hyperparameters.
    """
    n_layers: int
    n_neurons: int
    config: DefaultConfig
    out_seq_len: int
    inp_dim: int      # this is number of niche genes

    @nn.compact
    def __call__(self, inputs, deterministic = False, dropout_rng = random.key(0)):

        config = self.config

        x = inputs#.astype('int32')

        # x = Multiplyer(config, self.out_seq_len)(x)   
        x = FeedForward(n_layers=self.n_layers, 
                        n_neurons=self.n_neurons, 
                        n_output=self.out_seq_len * config.emb_dim)(x)   # output dim according to emb_dim to get ready for attention layers
        x = jnp.reshape(x, [x.shape[0], self.out_seq_len, config.emb_dim])
        for _ in range(config.num_layers):
            x = DecoderBlock(config)(inputs = x, 
                                    deterministic = deterministic, 
                                    dropout_rng = dropout_rng)
        x = WormholeFeedForward(config)(x)
        x = Unembedding(config, self.inp_dim)(x)

        # do scaling
        output = (nn.sigmoid(x) * 2 - 1) / sqrt(self.inp_dim)

        return output
    

