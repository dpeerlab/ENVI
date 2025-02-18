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
import scipy.sparse

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
            dec_cov = self.decoder_cov(z_conf)
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


def CalcCovMats(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1):
    """
    :meta private:
    """

    if isinstance(spatial_data.X, np.ndarray):
        ExpData = np.log(spatial_data[:, genes].X + 1)
    else:
        ExpData = np.log(np.asarray(spatial_data[:, genes].X.todense()) + 1)

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
            if 'log' in spatial_data.layers.keys():
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
            elif('log1p' in spatial_data.layers.keys()):
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log1p")
            elif(spatial_data.X.min() < 0):
                sc.pp.highly_variable_genes(spatial_data, n_top_genes=g)
            else:
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
