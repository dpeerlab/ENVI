# ENVI - Integrating scRNA-seq and spatial data to impute missing genes and reconstruct spatial context

![alt text](envi_schem.png?raw=true)

# Installation

Install ENVI (and COVET) through pypi:

```
pip install scENVI

```

# Tutorial

For a tutorial on how to run ENVI and analyze the results, go to example notebook at: https://github.com/dpeerlab/ENVI/blob/main/MOp_MERFISH_tutorial.ipynb.

# Usage

To run ENVI:

```
from scENVI import ENVI

ENVI_Model = ENVI.ENVI(spatial_data = st_data, sc_data = sc_data)
ENVI_Model.train()
ENVI_Model.impute()
ENVI_Model.infer_cov()

```

and to get the model's outputs:

```
st_data.obsm['envi_latent'] = ENVI_Model.spatial_data.obsm['envi_latent']
st_data.obsm['COVET'] = ENVI_Model.spatial_data.obsm['COVET']
st_data.obsm['COVET_SQRT'] = ENVI_Model.spatial_data.obsm['COVET_SQRT']
st_data.uns['COVET_genes'] =  ENVI_Model.covet_genes

st_data.obsm['imputation'] = ENVI_Model.spatial_data.obsm['imputation']


sc_data.obsm['envi_latent'] = ENVI_Model.sc_data.obsm['envi_latent']
sc_data.obsm['COVET'] = ENVI_Model.sc_data.obsm['COVET']
sc_data.obsm['COVET_SQRT'] = ENVI_Model.sc_data.obsm['COVET_SQRT']
sc_data.uns['COVET_genes'] =  ENVI_Model.covet_genes

```

And To run COVET (just on spatial data):

```
ENVI.COVET(st_data, k = 8, g = 64, spatial_key = 'spatial')
```

COVET information will be in:

```
st_data.obsm['COVET']
st_data.obsm['COVET_SQRT']
st_data.uns['COVET_Genes']
```
