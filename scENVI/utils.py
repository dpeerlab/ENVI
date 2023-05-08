import os
import tensorflow as tf
import numpy as np
import sklearn.neighbors
import scipy.sparse
import scanpy as sc
import scipy.special
import tensorflow_probability as tfp
import sklearn.neural_network
import scipy.sparse
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()


def MatSqrtTF(Mats):
    """
    Computes pseudo matrix square root with tensorflow linear algebra on cpu

    Args:
        Mats (array): Matrices to compute square root of
    Return:
        SqrtMats (np.array): pseudo matrix square of Mats
    """
    with tf.device("/CPU:0"):
        e, v = tf.linalg.eigh(Mats)
        e = tf.where(e < 0, 0, e)
        e = tf.math.sqrt(e)
        return tf.linalg.matmul(
            tf.linalg.matmul(v, tf.linalg.diag(e)), v, transpose_b=True
        ).numpy()


def BatchKNN(data, batch, k):
    """
    Computes kNN matrix for spatial data from multiple batches

    Args:
        data (array): Data to compute kNN on
        batch (array): Batch allocation per sample in Data
        k (int): number of neighbors for kNN matrix
    Return:
        kNNGraphIndex (np.array): for each sample, the index of its k nearest-neighbors
        WeightedIndex (np.array): Weighted (softmax) distance to each nearest-neighbors
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))
    WeightedIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )

        batch_knn_weight = scipy.special.softmax(
            -np.reshape(batch_knn.data, [data[val_ind].shape[0], k]), axis=-1
        )

        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]
        WeightedIndex[val_ind] = batch_knn_weight
    return (kNNGraphIndex.astype("int"), WeightedIndex)


def GetNeighExp(spatial_data, kNN, spatial_key="spatial", batch_key=-1, data_key=None):
    """
    Computing Niche mean expression based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        data_key (str): obsm key to compute niche mean across
            (default None, uses gene expression .X)

    Return:
        NeighExp: Average gene expression in niche
        kNNGraphIndex: indices of nearest spatial neighbors per cell
    """

    if data_key is None:
        Data = spatial_data.X
    else:
        Data = spatial_data.obsm[data_key]

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=kNN, mode="distance", n_jobs=-1
        ).tocoo()
        kNNGraph = scipy.sparse.coo_matrix(
            (np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)),
            shape=kNNGraph.shape,
        )
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex, _ = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    return Data[kNNGraphIndex[np.arange(spatial_data.obsm[spatial_key].shape[0])]]


def GetCOVET(
    spatial_data,
    kNN,
    spatial_key="spatial",
    batch_key=-1,
    MeanExp=None,
    weighted=False,
    cov_pc=1,
):
    """
    Wrapper to compute niche covariance based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        MeanExp (np.array): expression vector to shift niche covariance with
        weighted (bool): if True, weights covariance by spatial distance
    Return:
        COVET: niche covariance matrices
        kNNGraphIndex: indices of nearest spatial neighbors per cell
    """
    ExpData = spatial_data[:, spatial_data.var.highly_variable].X

    if cov_pc > 0:
        ExpData = np.log(ExpData + cov_pc)

    if batch_key == -1 or batch_key not in spatial_data.obs.columns:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=kNN, mode="distance", n_jobs=-1
        ).tocoo()
        kNNGraph = scipy.sparse.coo_matrix(
            (np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)),
            shape=kNNGraph.shape,
        )
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
        WeightedIndex = scipy.special.softmax(
            -np.reshape(kNNGraph.data, [spatial_data.obsm[spatial_key].shape[0], kNN]),
            axis=-1,
        )
    else:
        kNNGraphIndex, WeightedIndex = BatchKNN(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    if not weighted:
        WeightedIndex = np.ones_like(WeightedIndex) / kNN

    if MeanExp is None:
        DistanceMatWeighted = (
            (
                ExpData.mean(axis=0)[None, None, :]
                - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]
            )
            * np.sqrt(WeightedIndex)[:, :, None]
            * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis=-1)))[
                :, None, None
            ]
        )
    else:
        DistanceMatWeighted = (
            (MeanExp[:, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]])
            * np.sqrt(WeightedIndex)[:, :, None]
            * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis=-1)))[
                :, None, None
            ]
        )

    COVET = np.matmul(DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted)
    COVET = COVET + COVET.mean() * 0.00001 * np.expand_dims(
        np.identity(COVET.shape[-1]), axis=0
    )
    return (COVET, kNNGraphIndex)


def GetCov(
    spatial_data, k, g, genes, cov_dist, spatial_key="spatial", batch_key=-1, cov_pc=1
):
    """
    Compte niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        g (int): number of HVG to compute niche covariance matrices
        genes (list of str): list of genes to keep for niche covariance
        cov_dist (str): distribution to transform niche covariance matrices to fit into
        batch_key (str): obs key for batch information (default -1, for no batch)

    Return:
        COVET: raw, untransformed niche covariance matrices
        COVET_SQRT: covariance matrices transformed into chosen cov_dist
        NeighExp: Average gene expression in niche
        CovGenes: Genes used for niche covariance
    """

    spatial_data.layers["log"] = np.log(spatial_data.X + 1)

    if g == -1:
        CovGeneSet = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
        if g == 0:
            spatial_data.var.highly_variable = False
        if len(genes) > 0:
            spatial_data.var["highly_variable"][genes] = True

    CovGeneSet = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    CovGenes = spatial_data.var_names[CovGeneSet]

    COVET, kNNGraphIndex = GetCOVET(
        spatial_data,
        k,
        spatial_key=spatial_key,
        batch_key=batch_key,
        weighted=False,
        cov_pc=cov_pc,
    )
    NicheMat = spatial_data.X[kNNGraphIndex[np.arange(spatial_data.shape[0])]]

    if cov_dist == "norm":
        COVET_SQRT = COVET.reshape([COVET.shape[0], -1])
        COVET_SQRT = (
            COVET_SQRT - COVET_SQRT.mean(axis=0, keepdims=True)
        ) / COVET_SQRT.std(axis=0, keepdims=True)
    if cov_dist == "OT":
        COVET_SQRT = MatSqrtTF(COVET)
    else:
        COVET_SQRT = np.copy(COVET)

    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        NicheMat.astype("float32"),
        CovGenes,
    )


def LogNormalKL(mean, log_std, agg=None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if agg is None:
        return KL
    if not isinstance(agg, (str)):
        return tf.reduce_mean(KL, axis=-1)
    if agg == "sum":
        return tf.reduce_sum(KL, axis=-1)
    return tf.reduce_mean(KL, axis=-1)


def NormalKL(mean, log_std, agg=None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if agg is None:
        return KL
    if not isinstance(agg, (str)):
        return tf.reduce_mean(KL, axis=-1)
    if agg == "sum":
        return tf.reduce_sum(KL, axis=-1)

    return tf.reduce_mean(KL, axis=-1)


def log_pos_pdf(sample, rate, agg=None):
    log_prob = tfp.distributions.Poisson(rate=rate).log_prob(sample)
    if agg is None:
        return log_prob
    if not isinstance(agg, (str)):
        return tf.reduce_mean(log_prob * agg[None, : log_prob.shape[-1]], axis=-1)
    if agg == "sum":
        return tf.reduce_sum(log_prob, axis=-1)
    return tf.reduce_mean(log_prob, axis=-1)


def log_nb_pdf(sample, r, p, agg=None):
    log_prob = tfp.distributions.NegativeBinomial(total_count=r, logits=p).log_prob(
        sample
    )
    if agg is None:
        return log_prob
    if not isinstance(agg, (str)):
        return tf.reduce_mean(log_prob * agg[None, : log_prob.shape[-1]], axis=-1)
    if agg == "sum":
        return tf.reduce_sum(log_prob, axis=-1)
    return tf.reduce_mean(log_prob, axis=-1)


def log_zinb_pdf(sample, r, p, d, agg=None):
    log_prob = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=tf.stack([d, 1 - d], -1)),
        components=[
            tfp.distributions.Deterministic(loc=tf.zeros_like(d)),
            tfp.distributions.NegativeBinomial(total_count=r, logits=p),
        ],
    ).log_prob(sample)

    if agg is None:
        return log_prob
    if not isinstance(agg, (str)):
        return tf.reduce_mean(log_prob * agg[None, : log_prob.shape[-1]], axis=-1)
    if agg == "sum":
        return tf.reduce_sum(log_prob, axis=-1)
    return tf.reduce_mean(log_prob, axis=-1)


def OTDistance(sample, mean, agg=None):
    sample = tf.reshape(sample, [sample.shape[0], -1])
    mean = tf.reshape(mean, [mean.shape[0], -1])
    log_prob = -tf.square(sample - mean)
    if agg is None:
        return log_prob
    if not isinstance(agg, (str)):
        return tf.reduce_mean(log_prob, axis=-1)
    if agg == "sum":
        return tf.reduce_sum(log_prob, axis=-1)
    return tf.reduce_mean(log_prob, axis=-1)


def log_normal_pdf(sample, mean, scale, agg=None):
    log_prob = tfp.distributions.Normal(loc=mean, scale=tf.exp(scale)).log_prob(sample)
    if agg is None:
        return log_prob
    if not isinstance(agg, (str)):
        return tf.reduce_mean(log_prob * agg[None, : log_prob.shape[-1]], axis=-1)
    if agg == "sum":
        return tf.reduce_sum(log_prob, axis=-1)
    return tf.reduce_mean(log_prob, axis=-1)


@tf.function
def trace_log(Mat):
    return tf.reduce_mean(tf.math.log(tf.linalg.diag_part(Mat)), axis=-1)


@tf.function
def log_wish_pdf(sample, scale, agg="mean"):
    if agg == "mean":
        return tfp.distributions.WishartTriL(
            df=sample.shape[-1], scale_tril=scale, input_output_cholesky=True
        ).log_prob(sample) / (sample.shape[-1] ** 2)
    elif agg == "mean":
        return tfp.distributions.WishartTriL(
            df=sample.shape[-1], scale_tril=scale, input_output_cholesky=True
        ).log_prob(sample)
