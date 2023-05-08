import anndata
import numpy as np
import pytest
import scanpy as sc

from scENVI.utils import (
    batch_knn,
    get_niche_covariance,
    get_covet,
    get_niche_expression,
    matrix_square_root,
)


@pytest.fixture
def sample_spatial_data():
    obs = np.random.randn(100, 2)
    var = np.random.poisson(lam=10, size=(100, 50))
    spatial_data = anndata.AnnData(X=var, obsm={"spatial": obs})
    return spatial_data


def test_matrix_square_root():
    mats = np.random.randn(5, 3, 3)
    sqrt_mats = matrix_square_root(mats)
    assert sqrt_mats.shape == mats.shape


def test_batch_knn(sample_spatial_data):
    batch = np.random.randint(0, 2, size=sample_spatial_data.shape[0])
    k = 5
    knn_graph_index, weighted_index = batch_knn(
        sample_spatial_data.obsm["spatial"], batch, k
    )
    assert knn_graph_index.shape == (sample_spatial_data.shape[0], k)
    assert weighted_index.shape == (sample_spatial_data.shape[0], k)


def test_get_niche_expression(sample_spatial_data):
    kNN = 5
    neigh_exp = get_niche_expression(sample_spatial_data, kNN)
    assert neigh_exp.shape == (
        sample_spatial_data.shape[0],
        kNN,
        sample_spatial_data.shape[1],
    )


def test_get_covet(sample_spatial_data):
    kNN = 5
    n_top_genes = 10
    sc.pp.highly_variable_genes(sample_spatial_data, n_top_genes=n_top_genes)
    covet, knn_graph_index = get_covet(sample_spatial_data, kNN)
    assert covet.shape == (sample_spatial_data.shape[0], n_top_genes, n_top_genes)
    assert knn_graph_index.shape == (sample_spatial_data.shape[0], kNN)


def test_get_niche_covariance(sample_spatial_data):
    k = 5
    g = 20
    genes = []
    n_cells, n_genes = sample_spatial_data.shape
    cov_dist = "norm"
    covet, covet_sqrt, niche_mat, cov_genes = get_niche_covariance(
        sample_spatial_data, k, g, genes, cov_dist
    )
    assert covet.shape == (n_cells, g, g)
    assert covet_sqrt.shape == (n_cells, g, g)
    assert niche_mat.shape == (n_cells, k, n_genes)
    assert cov_genes.shape[0] == g
