import pytest

import anndata
import numpy as np

import scenvi

@pytest.fixture
def example_model():
    st_data = anndata.AnnData(X = np.random.uniform(low = 0, high = 100, size = (16,4)), obsm = {'spatial': np.random.normal(size = [16,2])})
    sc_data = anndata.AnnData(X = np.random.uniform(low = 0, high = 100, size = (16,8)))
    
    envi_model = scenvi.ENVI(spatial_data = st_data, sc_data = sc_data, batch_key = -1)
    return(envi_model)

def test_train(example_model):
    example_model.train(training_steps = 1)
    

    assert 'COVET' in  example_model.spatial_data.obsm
    assert 'COVET_SQRT' in  example_model.spatial_data.obsm
    
    assert 'envi_latent' in  example_model.spatial_data.obsm
    assert example_model.spatial_data.obsm['envi_latent'].shape == (example_model.spatial_data.shape[0], example_model.latent_dim)
        
    assert 'envi_latent' in  example_model.sc_data.obsm
    assert example_model.sc_data.obsm['envi_latent'].shape == (example_model.sc_data.shape[0], example_model.latent_dim)
    
    
def test_impute(example_model):
    example_model.train(training_steps = 1)
    example_model.impute_genes()

    assert 'imputation' in  example_model.spatial_data.obsm
        
def test_infer_niche(example_model):
    example_model.train(training_steps = 1)
    example_model.infer_niche_covet()
    
    assert 'COVET_SQRT' in  example_model.sc_data.obsm
    assert 'COVET' in  example_model.sc_data.obsm
    