PyENVI for Python3
======================


ENVI is a deep learnining based variational inference method to integrate scRNAseq with spatial sequencing data. 
It creates a combined latent space for both data modalities, from which missing gene can be imputed from spatial data, cell types labels can be transfered
and the spatial niche can be reconstructed for the dissociated scRNAseq data

This implementation is written in Python3 and relies on jax, flax, sklearn, scipy and scanpy.  


To install JAX, simply run the command:

    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

And to install ENVI along with the rest of the requirements: 

    pip install scenvi

To run ENVI:

    import scenvi 
    
    envi_model = scenvi.ENVI(spatial_data = st_data, sc_data = sc_data)
    
    envi_model.train()
    envi_model.impute_genes()
    envi_model.infer_niche()
     
    
    st_data.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
    st_data.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
    st_data.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
    st_data.uns['COVET_genes'] =  envi_model.CovGenes
    st_data.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']

    sc_data.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
    sc_data.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
    sc_data.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
    sc_data.uns['COVET_genes'] =  envi_model.CovGenes
    
And to just compute COVET for spatial data:


    st_data.obsm['COVET'], st_data.obsm['COVET_SQRT'], st_data.uns['CovGenes'] = scenvi.compute_covet(st_data)
        
