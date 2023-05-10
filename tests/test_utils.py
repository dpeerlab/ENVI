import anndata
import numpy as np
import pytest
import scanpy as sc

from utils import (
    batch_knn,
    compute_covet,
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
    k = 5
    n_cells, n_genes = sample_spatial_data.shape
    neigh_exp = get_niche_expression(sample_spatial_data, k)
    assert neigh_exp.shape == (n_cells, k, n_genes)


def test_compute_covet(sample_spatial_data):
    k = 5
    n_top_genes = 10
    sc.pp.highly_variable_genes(sample_spatial_data, n_top_genes=n_top_genes)
    n_top_genes = sample_spatial_data.var[
        "highly_variable"
    ].sum()  # may be different than n_top_genes
    covet, knn_graph_index = compute_covet(sample_spatial_data, k)
    assert covet.shape == (sample_spatial_data.shape[0], n_top_genes, n_top_genes)
    assert knn_graph_index.shape == (sample_spatial_data.shape[0], k)


def test_get_covet(sample_spatial_data):
    k = 5
    g = 20
    genes = []
    n_cells, n_genes = sample_spatial_data.shape
    covet_distribution = "norm"
    covet, covet_sqrt, niche_mat, covet_genes = get_covet(
        sample_spatial_data, k, g, genes, covet_distribution
    )
    assert covet.shape == (n_cells, g, g)
    assert covet_sqrt.shape == (n_cells, g, g)
    assert niche_mat.shape == (n_cells, k, n_genes)
    assert covet_genes.shape[0] == g
