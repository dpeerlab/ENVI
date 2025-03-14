import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import random

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm   

class FeedForward(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_output: int

    @nn.compact
    def __call__(self, x):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_output = self.n_output

        x = nn.Dense(
            features=n_neurons,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x)

        for _ in range(n_layers - 1):

            x = nn.Dense(
                features=n_neurons,
                dtype=jnp.float32,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = nn.leaky_relu(x) + x
            x = nn.LayerNorm(dtype=jnp.float32)(x)

        output = nn.Dense(
            features=n_output,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)

        return output


class CVAE(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_latent: int
    n_output_exp: int
    n_output_cov: int

    def setup(self):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_latent = self.n_latent
        n_output_exp = self.n_output_exp
        n_output_cov = self.n_output_cov

        self.encoder = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_latent * 2
        )

        self.decoder_exp = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_exp
        )

        self.decoder_cov = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_cov
        )

    def __call__(self, x, mode="spatial", key=random.key(0)):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)
        enc_mu, enc_logstd = jnp.split(self.encoder(x_conf), 2, axis=-1)

        key, subkey = random.split(key)
        z = enc_mu + random.normal(key=subkey, shape=enc_logstd.shape) * jnp.exp(
            enc_logstd
        )
        z_conf = jnp.concatenate([z, conf_neurons], axis=-1)

        dec_exp = self.decoder_exp(z_conf)

        if mode == "spatial":
            dec_cov = self.decoder_cov(z)
            return (enc_mu, enc_logstd, dec_exp, dec_cov)
        return (enc_mu, enc_logstd, dec_exp)


@struct.dataclass
class Metrics(metrics.Collection):
    """
    :meta private:
    """

    enc_loss: metrics.Average
    dec_loss: metrics.Average
    enc_corr: metrics.Average


class TrainState(train_state.TrainState):
    """
    :meta private:
    """

    metrics: Metrics

def batch_matrix_sqrt(Mats):
    """
    :meta private:
    """

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def batch_knn(data, batch, k):
    """
    :meta private:
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def calculate_covariance_matrices(spatial_data, kNN, genes, spatial_key="spatial", batch_key=-1, batch_size=None):
    """
    :meta private:
    """
    
    ExpData = np.log(spatial_data[:, genes].X + 1)

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        kNNGraphIndex = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        kNNGraphIndex = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )
    
    # Get the global mean for each gene (used as reference point)
    global_mean = ExpData.mean(axis=0)
    
    # Initialize the output covariance matrices
    n_cells = ExpData.shape[0]
    n_genes = len(genes)
    CovMats = np.zeros((n_cells, n_genes, n_genes), dtype=np.float32)
    
    # Process in batches if requested
    if batch_size is None or batch_size >= n_cells:
        # Process all cells at once (original implementation)
        print("Calculating covariance matrices for all cells/spots")
        DistanceMatWeighted = (
            global_mean[None, None, :]
            - ExpData[kNNGraphIndex[np.arange(n_cells)]]
        )
        
        CovMats = np.matmul(
            DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted
        ) / (kNN - 1)
    else:
        
        # Process in batches to save memory
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        
        for batch_idx in tqdm(batch_indices, desc="Calculating covariance matrices"):
            # Get neighbor indices for this batch
            batch_neighbors = kNNGraphIndex[batch_idx]
            
            # Calculate the distance matrices for this batch
            batch_distances = (
                global_mean[None, None, :]
                - ExpData[batch_neighbors]
            )
            
            # Calculate the covariance matrices for this batch
            batch_covs = np.matmul(
                batch_distances.transpose([0, 2, 1]), batch_distances
            ) / (kNN - 1)
            
            # Store the results
            CovMats[batch_idx] = batch_covs
    
    # Add a small regularization term to ensure positive definiteness
    reg_term = CovMats.mean() * 0.00001
    identity = np.eye(n_genes)[None, :, :]
    CovMats = CovMats + reg_term * identity
    
    return CovMats

def niche_cell_type(
    spatial_data, kNN, spatial_key="spatial", cell_type_key="cell_type", batch_key=-1
):
    """
    :meta private:
    """

    

    if batch_key == -1:
        kNNGraph = sklearn.neighbors.kneighbors_graph(
            spatial_data.obsm[spatial_key],
            n_neighbors=kNN,
            mode="connectivity",
            n_jobs=-1,
        ).tocoo()
        knn_index = np.reshape(
            np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN]
        )
    else:
        knn_index = batch_knn(
            spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN
        )

    one_hot_enc = OneHotEncoder().fit(
        np.asarray(list(set(spatial_data.obs[cell_type_key]))).reshape([-1, 1])
    )
    cell_type_one_hot = (
        one_hot_enc.transform(
            np.asarray(spatial_data.obs[cell_type_key]).reshape([-1, 1])
        )
        .reshape([spatial_data.obs["cell_type"].shape[0], -1])
        .todense()
    )

    cell_type_niche = pd.DataFrame(
        cell_type_one_hot[knn_index].sum(axis=1),
        index=spatial_data.obs_names,
        columns=list(one_hot_enc.categories_[0]),
    )
    return cell_type_niche

def compute_covet(
    spatial_data, k=8, g=64, genes=None, spatial_key="spatial", batch_key="batch", batch_size=None
):
    """
    Compute niche covariance matrices for spatial data, run with scenvi.compute_covet

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param g: (int) number of HVG to compute COVET representation on (default 64)
    :param genes: (list of str) list of genes to keep for niche covariance (default [])
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)
    :batch_size : (int)  Number of cells/spots to process at once for large datasets (default None)
        
    :return COVET: niche covariance matrices
    :return COVET_SQRT: matrix square-root of niche covariance matrices for approximate OT
    :return CovGenes: list of genes selected for COVET representation
    """

    genes = [] if genes is None else genes
    
    # Handle batch key
    if batch_key not in spatial_data.obs.columns:
        batch_key = -1
    
    # Select genes for covariance calculation
    if g == -1:
        # Use all genes
        CovGenes = spatial_data.var_names
        print(f"Computing COVET using all {len(CovGenes)} genes")
    else:
        # Check if highly variable genes need to be calculated
        if "highly_variable" not in spatial_data.var.columns:
            print(f"Identifying top {g} highly variable genes for COVET calculation")
            # Create a copy to avoid modifying the input
            spatial_data_copy = spatial_data.copy()
            
            # Determine appropriate layer for HVG calculation
            if 'log' in spatial_data_copy.layers:
                layer = "log"
            elif 'log1p' in spatial_data_copy.layers:
                layer = "log1p"
            elif spatial_data_copy.X.min() < 0:
                # Data is already log-transformed
                layer = None
            else:
                # Create log layer
                spatial_data_copy.layers["log"] = np.log(spatial_data_copy.X + 1)
                layer = "log"
            
            # Calculate HVGs
            sc.pp.highly_variable_genes(
                spatial_data_copy, 
                n_top_genes=g, 
                layer=layer if layer else None
            )
            
            # Get HVG names
            hvg_genes = spatial_data_copy.var_names[spatial_data_copy.var.highly_variable]
        else:
            # HVGs already calculated
            hvg_genes = spatial_data.var_names[spatial_data.var.highly_variable]
            print(f"Using {len(hvg_genes)} pre-calculated highly variable genes for COVET")
        
        # Combine HVGs with manually specified genes
        CovGenes = np.asarray(hvg_genes)
        if len(genes) > 0:
            CovGenes = np.union1d(CovGenes, genes)
            print(f"Added {len(genes)} user-specified genes to COVET calculation")
            
    
    # Calculate covariance matrices with batch processing
    COVET = calculate_covariance_matrices(
        spatial_data, k, genes=CovGenes, spatial_key=spatial_key, 
        batch_key=batch_key, batch_size=batch_size
    )
    
    # Calculate matrix square root

    if batch_size is None or batch_size >= COVET.shape[0]:
        print("Computing matrix square root...")
        COVET_SQRT = batch_matrix_sqrt(COVET)
    else:
        # Process matrix square root in batches too
        n_cells = COVET.shape[0]
        COVET_SQRT = np.zeros_like(COVET)
        
        # Split into batches
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        
        for batch_idx in tqdm(batch_indices, desc="Computing matrix square roots"):
            # Process this batch of matrices
            batch_sqrt = batch_matrix_sqrt(COVET[batch_idx])
            COVET_SQRT[batch_idx] = batch_sqrt
    
    # Return results with proper types
    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes, dtype=str),
    )