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

envi_model = ENVI.ENVI(spatial_data = st_data, sc_data = sc_data)
envi_model.train()
envi_model.impute()
envi_model.infer_covet()

```

and to get the model's outputs:

```
st_data.obsm['envi_latent'] = envi_model.spatial_data.obsm['envi_latent']
st_data.obsm['COVET'] = envi_model.spatial_data.obsm['COVET']
st_data.obsm['COVET_SQRT'] = envi_model.spatial_data.obsm['COVET_SQRT']
st_data.uns['COVET_genes'] =  envi_model.covet_genes

st_data.obsm['imputation'] = envi_model.spatial_data.obsm['imputation']


sc_data.obsm['envi_latent'] = envi_model.sc_data.obsm['envi_latent']
sc_data.obsm['COVET'] = envi_model.sc_data.obsm['COVET']
sc_data.obsm['COVET_SQRT'] = envi_model.sc_data.obsm['COVET_SQRT']
sc_data.uns['COVET_genes'] =  envi_model.covet_genes

```

And To run COVET (just on spatial data):

```
ENVI.covet(st_data, k=8, g=64, spatial_key='spatial')
```

COVET information will be in:

```
st_data.obsm['COVET']
st_data.obsm['COVET_SQRT']
st_data.uns['COVET_Genes']
```
