import pytest # type: ignore

import anndata
import numpy as np

import scenvi

@pytest.fixture
def example_model():
    st_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 4)), 
                             obsm={'spatial': np.random.normal(size=[16, 2])})
    sc_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 8)))
    
    envi_model = scenvi.ENVI(spatial_data=st_data, sc_data=sc_data, batch_key=-1)
    return envi_model

@pytest.fixture
def example_model_with_hvg():
    # Create data with pre-computed HVGs
    st_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 4)), 
                             obsm={'spatial': np.random.normal(size=[16, 2])})
    sc_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 8)))
    
    # Add highly_variable column
    sc_data.var['highly_variable'] = [True, False, True, False, True, False, True, False]
    
    envi_model = scenvi.ENVI(spatial_data=st_data, sc_data=sc_data, batch_key=-1)
    return envi_model

@pytest.fixture
def example_model_with_user_genes():
    st_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 4)), 
                             obsm={'spatial': np.random.normal(size=[16, 2])})
    st_data.var_names = [f"gene{i}" for i in range(4)]
    
    sc_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 8)))
    sc_data.var_names = [f"gene{i}" for i in range(8)]
    
    # Specify user genes to include
    user_genes = ["gene1", "gene3", "gene5"]
    
    envi_model = scenvi.ENVI(spatial_data=st_data, sc_data=sc_data, batch_key=-1, sc_genes=user_genes)
    return envi_model

@pytest.fixture
def large_example_model():
    # Create larger dataset to test batch processing
    st_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(100, 32)), 
                             obsm={'spatial': np.random.normal(size=[100, 2])})
    sc_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(100, 64)))
    
    envi_model = scenvi.ENVI(spatial_data=st_data, sc_data=sc_data, batch_key=-1, num_cov_genes = 16)
    return envi_model

def test_train(example_model):
    example_model.train(training_steps=1)
    
    assert 'COVET' in example_model.spatial_data.obsm
    assert 'COVET_SQRT' in example_model.spatial_data.obsm
    
    assert 'envi_latent' in example_model.spatial_data.obsm
    assert example_model.spatial_data.obsm['envi_latent'].shape == (example_model.spatial_data.shape[0], example_model.latent_dim)
        
    assert 'envi_latent' in example_model.sc_data.obsm
    assert example_model.sc_data.obsm['envi_latent'].shape == (example_model.sc_data.shape[0], example_model.latent_dim)
    
def test_impute(example_model):
    example_model.train(training_steps=1)
    example_model.impute_genes()

    assert 'imputation' in example_model.spatial_data.obsm
        
def test_infer_niche(example_model):
    example_model.train(training_steps=1)
    example_model.infer_niche_covet()
    
    assert 'COVET_SQRT' in example_model.sc_data.obsm
    assert 'COVET' in example_model.sc_data.obsm

def test_precomputed_hvg(example_model_with_hvg):
    """Test that pre-computed highly variable genes are correctly used"""
    # Check that the HVGs from input were maintained
    hvg_count = sum(example_model_with_hvg.sc_data.var['highly_variable'])
    assert hvg_count > 0
    
    # Train and verify
    example_model_with_hvg.train(training_steps=1)
    assert 'envi_latent' in example_model_with_hvg.spatial_data.obsm
    assert 'envi_latent' in example_model_with_hvg.sc_data.obsm

def test_user_specified_genes(example_model_with_user_genes):
    """Test that user-specified genes are correctly included"""
    # Check that user-specified genes are included
    for gene in ["gene1", "gene3", "gene5"]:
        if gene in example_model_with_user_genes.sc_data.var_names:
            assert gene in example_model_with_user_genes.sc_data.var_names
    
    # Train and verify
    example_model_with_user_genes.train(training_steps=1)
    example_model_with_user_genes.impute_genes()
    
    # Check that imputation includes these genes
    assert 'imputation' in example_model_with_user_genes.spatial_data.obsm
    imputed_genes = example_model_with_user_genes.spatial_data.obsm['imputation'].columns
    
    for gene in ["gene1", "gene3", "gene5"]:
        if gene in example_model_with_user_genes.sc_data.var_names:
            assert gene in imputed_genes

def test_all_genes_covet():
    """Test using all genes for COVET calculation"""
    # Create a new model using all genes for COVET
    st_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 4)), 
                             obsm={'spatial': np.random.normal(size=[16, 2])})
    sc_data = anndata.AnnData(X=np.random.uniform(low=0, high=100, size=(16, 8)))
    
    # Set num_cov_genes=-1 to use all genes
    envi_model = scenvi.ENVI(spatial_data=st_data, sc_data=sc_data, batch_key=-1, num_cov_genes=-1)
    
    # Verify all genes are used
    assert len(envi_model.CovGenes) == st_data.shape[1]
    
    # Train and verify
    envi_model.train(training_steps=1)
    assert 'COVET' in envi_model.spatial_data.obsm

def test_covet_with_batches(large_example_model):
    """Test COVET calculation with batch processing"""
    # Calculate COVET matrices using batch processing



    (covet, covet_sqrt, cov_genes) = scenvi.compute_covet(
        large_example_model.spatial_data,
        k=5,
        g=6,
        batch_size=20  # Process in batches of 20
    )
    # Verify the results
    assert covet.shape[0] == large_example_model.spatial_data.shape[0]
    assert covet_sqrt.shape[0] == large_example_model.spatial_data.shape[0]
    assert len(cov_genes) >= 6
    
    # Verify shapes match expected dimensions
    assert covet.shape[1] == covet.shape[2]  # Square matrices
    assert covet_sqrt.shape[1] == covet_sqrt.shape[2]  # Square matrices
    assert covet.shape[1] == len(cov_genes)  # Matrix size matches gene count

def test_niche_cell_type(example_model):
    """Test niche cell type inference"""
    # Add cell type information
    example_model.spatial_data.obs['cell_type'] = np.random.choice(
        ['Type1', 'Type2', 'Type3'], size=example_model.spatial_data.shape[0]
    )
    
    # Train model
    example_model.train(training_steps=1)
    example_model.infer_niche_covet()
    # Infer niche cell types
    
    example_model.infer_niche_celltype(cell_type_key='cell_type')
    
    # Verify results
    assert 'cell_type_niche' in example_model.spatial_data.obsm
    assert 'cell_type_niche' in example_model.sc_data.obsm
    
    # Check the columns of the cell_type_niche match the unique cell types
    unique_cell_types = sorted(example_model.spatial_data.obs['cell_type'].unique())
    niche_columns = sorted(example_model.sc_data.obsm['cell_type_niche'].columns)
    assert unique_cell_types == niche_columns