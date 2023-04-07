# ENVI - Integrating scRNA-seq and spatial data to impute missing genes and reconstruct spatial context 

![alt text](img/envi_schem.png?raw=true)

# Installation 
Insatll ENVI (and COVET) through pypi:
 
```
!python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps scENVI

```

# Tutorial

For a notebook on how to run ENVI and analyze the results, go to example notebook.


# Usage

To run ENVI:
```
from scENVI import ENVI

model = ENVI.ENVI(spatial_data = st_data, sc_data = sc_data)
model.Train()
ENVI_Model.impute()
ENVI_Model.infer_cov()

```
 
and to get the model's outputs:


```
st_data.obsm['envi_latent'] = ENVI_Model.spatial_data.obsm['envi_latent']
st_data.obsm['COVET'] = ENVI_Model.spatial_data.obsm['COVET']
st_data.obsm['COVET_SQRT'] = ENVI_Model.spatial_data.obsm['COVET_SQRT']
st_data.uns['COVET_genes'] =  ENVI_Model.CovGenes

st_data.obsm['imputation'] = ENVI_Model.spatial_data.obsm['imputation']


sc_data.obsm['envi_latent'] = ENVI_Model.sc_data.obsm['envi_latent']
sc_data.obsm['COVET'] = ENVI_Model.sc_data.obsm['COVET']
sc_data.obsm['COVET_SQRT'] = ENVI_Model.sc_data.obsm['COVET_SQRT']
sc_data.uns['COVET_genes'] =  ENVI_Model.CovGenes

```
And To run COVET (just on spatial data):

```
ENVI.COVET(st_data, k = 8, g = 64,spatial_key = 'spatial')
```

COVET information will be in:

```
st_data.obsm['COVET'] 
st_data.obsm['COVET_SQRT'] 
st_data.uns['COVET_Genes']
```


