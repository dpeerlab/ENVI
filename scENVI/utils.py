import os
import sys

import numpy as np
import scanpy as sc
import scipy.sparse
import scipy.special
import sklearn.neighbors
import sklearn.neural_network
import tensorflow as tf
import tensorflow_probability as tfp


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()


def matrix_square_root(matrices):
    """
    Computes pseudo matrix square root with tensorflow linear algebra on cpu

    Args:
        matrices (array): Matrices to compute square root of
    Return:
        pseudo matrix square roots of matrices
    """
    with tf.device("/CPU:0"):
        e, v = tf.linalg.eigh(matrices)
        e = tf.where(e < 0, 0, e)
        e = tf.math.sqrt(e)
        return tf.linalg.matmul(
            tf.linalg.matmul(v, tf.linalg.diag(e)), v, transpose_b=True
        ).numpy()


def batch_knn(data, batch, k):
    """
    Computes kNN matrix for spatial data from multiple batches

    Args:
        data (array): Data to compute kNN on
        batch (array): Batch allocation per sample in Data
        k (int): number of neighbors for kNN matrix
    Return:
        knn_graph_index (np.array): indices of each sample's k nearest-neighbors
        weighted_index (np.array): Weighted (softmax) distance to each nearest-neighbors
    """

    knn_graph_index = np.zeros(shape=(data.shape[0], k))
    weighted_index = np.zeros(shape=(data.shape[0], k))

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

        knn_graph_index[val_ind] = val_ind[batch_knn_ind]
        weighted_index[val_ind] = batch_knn_weight
    return (knn_graph_index.astype("int"), weighted_index)


def get_niche_expression(
    spatial_data, k, spatial_key="spatial", batch_key=-1, data_key=None
):
    """
    Computing Niche mean expression based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        data_key (str): obsm key to compute niche mean across
            (default None, uses gene expression .X)

    Return:
        niche_expression: Average gene expression in niche
        knn_graph_index: indices of nearest spatial neighbors per cell
    """

    if data_key is None:
        Data = spatial_data.X
    else:
        Data = spatial_data.obsm[data_key]

    if batch_key == -1:
        knn_graph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        knn_graph = scipy.sparse.coo_matrix(
            (np.ones_like(knn_graph.data), (knn_graph.row, knn_graph.col)),
            shape=knn_graph.shape,
        )
        knn_graph_index = np.reshape(
            np.asarray(knn_graph.col), [spatial_data.obsm[spatial_key].shape[0], k]
        )
    else:
        knn_graph_index, _ = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], k
        )

    return Data[knn_graph_index[np.arange(spatial_data.obsm[spatial_key].shape[0])]]


def get_covet(
    spatial_data,
    k,
    spatial_key="spatial",
    batch_key=-1,
    mean_expression=None,
    weighted=False,
    cov_pc=1,
):
    """
    Wrapper to compute niche covariance based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        mean_expression (np.array): expression vector to shift niche covariance with
        weighted (bool): if True, weights covariance by spatial distance
    Return:
        covet: niche covariance matrices
        knn_graph_index: indices of nearest spatial neighbors per cell
    """
    expression_data = spatial_data[:, spatial_data.var.highly_variable].X

    if cov_pc > 0:
        expression_data = np.log(expression_data + cov_pc)

    if batch_key == -1 or batch_key not in spatial_data.obs.columns:
        knn_graph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key], n_neighbors=k, mode="distance", n_jobs=-1
        ).tocoo()
        knn_graph = scipy.sparse.coo_matrix(
            (np.ones_like(knn_graph.data), (knn_graph.row, knn_graph.col)),
            shape=knn_graph.shape,
        )
        knn_graph_index = np.reshape(
            np.asarray(knn_graph.col), [spatial_data.obsm[spatial_key].shape[0], k]
        )
        weighted_index = scipy.special.softmax(
            -np.reshape(knn_graph.data, [spatial_data.obsm[spatial_key].shape[0], k]),
            axis=-1,
        )
    else:
        knn_graph_index, weighted_index = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], k
        )

    if not weighted:
        weighted_index = np.ones_like(weighted_index) / k

    if mean_expression is None:
        weighted_distance_matrix = (
            (
                expression_data.mean(axis=0)[None, None, :]
                - expression_data[knn_graph_index[np.arange(expression_data.shape[0])]]
            )
            * np.sqrt(weighted_index)[:, :, None]
            * np.sqrt(1 / (1 - np.sum(np.square(weighted_index), axis=-1)))[
                :, None, None
            ]
        )
    else:
        weighted_distance_matrix = (
            (
                mean_expression[:, None, :]
                - expression_data[knn_graph_index[np.arange(expression_data.shape[0])]]
            )
            * np.sqrt(weighted_index)[:, :, None]
            * np.sqrt(1 / (1 - np.sum(np.square(weighted_index), axis=-1)))[
                :, None, None
            ]
        )

    covet = np.matmul(
        weighted_distance_matrix.transpose([0, 2, 1]), weighted_distance_matrix
    )
    covet = covet + covet.mean() * 0.00001 * np.expand_dims(
        np.identity(covet.shape[-1]), axis=0
    )
    return (covet, knn_graph_index)


def get_niche_covariance(
    spatial_data, k, g, genes, cov_dist, spatial_key="spatial", batch_key=-1, cov_pc=1
):
    """
    Compute niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors to define niche
        g (int): number of HVG to compute niche covariance matrices
        genes (list of str): list of genes to keep for niche covariance
        cov_dist (str): distribution to transform niche covariance matrices to fit into
        batch_key (str): obs key for batch information (default -1, for no batch)

    Return:
        covet: raw, untransformed niche covariance matrices
        covet_sqrt: covariance matrices transformed into chosen cov_dist
        niche_expression: Average gene expression in niche
        covet_genes: Genes used for niche covariance
    """

    spatial_data.layers["log"] = np.log(spatial_data.X + 1)

    if g == -1:
        covet_gene_set = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.highly_variable_genes(spatial_data, n_top_genes=g, layer="log")
        if g == 0:
            spatial_data.var.highly_variable = False
        if len(genes) > 0:
            spatial_data.var["highly_variable"][genes] = True

    covet_gene_set = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    covet_genes = spatial_data.var_names[covet_gene_set]

    covet, knn_graph_index = get_covet(
        spatial_data,
        k,
        spatial_key=spatial_key,
        batch_key=batch_key,
        weighted=False,
        cov_pc=cov_pc,
    )
    niche_expression = spatial_data.X[knn_graph_index[np.arange(spatial_data.shape[0])]]

    if cov_dist == "norm":
        covet_sqrt = covet.reshape([covet.shape[0], -1])
        covet_sqrt = (
            covet_sqrt - covet_sqrt.mean(axis=0, keepdims=True)
        ) / covet_sqrt.std(axis=0, keepdims=True)
    if cov_dist == "OT":
        covet_sqrt = matrix_square_root(covet)
    else:
        covet_sqrt = np.copy(covet)

    return (
        covet.astype("float32"),
        covet_sqrt.astype("float32"),
        niche_expression.astype("float32"),
        covet_genes,
    )


def log_normal_kl(mean, log_std, agg=None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if agg is None:
        return KL
    if not isinstance(agg, (str)):
        return tf.reduce_mean(KL, axis=-1)
    if agg == "sum":
        return tf.reduce_sum(KL, axis=-1)
    return tf.reduce_mean(KL, axis=-1)


def normal_kl(mean, log_std, agg=None):
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


def ot_distance(sample, mean, agg=None):
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
