ENVI & COVET
======================


ENVI is a deep learnining based variational inference method to integrate scRNA-seq with spatial transcriptomics data. ENVI learns to reconstruct spatial onto for dissociated scRNA-seq data and impute unimagd genes onto spatial data.

This implementation is written in Python3 and relies on jax, flax, sklearn, scipy and scanpy.  


To install JAX, simply run the command:

    pip install -U "jax[cuda12]"

And to install ENVI along with the rest of the requirements: 

    pip install scenvi

To run ENVI:

    import scenvi 
    
    envi_model = scenvi.ENVI(spatial_data = st_data, sc_data = sc_data)
    
    envi_model.train()
    envi_model.impute_genes()
    envi_model.infer_niche_covet()
    envi_model.infer_niche_celltype()
    
    st_data.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
    st_data.uns['COVET_genes'] =  envi_model.CovGenes
    st_data.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
    st_data.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
    st_data.obsm['cell_type_niche'] = envi_model.spatial_data.obsm['cell_type_niche']
    st_data.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']


    sc_data.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
    sc_data.uns['COVET_genes'] =  envi_model.CovGenes
    sc_data.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
    sc_data.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
    sc_data.obsm['cell_type_niche'] = envi_model.sc_data.obsm['cell_type_niche']
    

And to just compute COVET for spatial data:

    st_data.obsm['COVET'], st_data.obsm['COVET_SQRT'], st_data.uns['CovGenes'] = scenvi.compute_covet(st_data)
        
Please read our documentation and see a full tutorial at https://scenvi.readthedocs.io/.
