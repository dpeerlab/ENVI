import os
import tensorflow as tf
import numpy as np
import sklearn.neighbors
import scipy.sparse
import anndata
import scanpy as sc
import scipy.special
import tensorflow_probability as tfp
import time
import sklearn.neural_network
import pandas as pd
import scipy.sparse
import pickle
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        

class LinearLayer(tf.keras.layers.Layer):
    """
    Costume keras linear layer

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
    """
    def __init__(self, units, input_dim, kernel_init, bias_init, name):
        super(LinearLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer=kernel_init, trainable=True, name = name + '/kernel')
        self.b = self.add_weight(shape=(units,), initializer=bias_init, trainable=True, name = name + '/bias')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class ConstantLayer(tf.keras.layers.Layer):
    """
    Costume keras constant layer, biases only

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        bias_init (keras initializer): initializer of neural biases
        comm_disp (bool): if True, spatial_dist and sc_dist share dispersion parameter(s)
        const_disp (bool): if True, dispertion parameter(s) are only per gene, rather there per gene per sample
    """
    def __init__(self, units, input_dim, bias_init, name):
        super(ConstantLayer, self).__init__()
        self.b = self.add_weight(shape=(units,), initializer=bias_init, trainable=True, name = name + '/bias')

    def call(self, inputs):
        return tf.tile(self.b[None, :], [inputs.shape[0], 1])
    
class ENVIOutputLayer(tf.keras.layers.Layer):
    """
    Costume keras layer for ENVI expression decoder output

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
        spatial_dist (str): distribution used to describe spatial data (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm') 
        sc_dist (str): distribution used to describe sinlge cell data (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
    """
    def __init__(self, 
                 input_dim, 
                 units, 
                 kernel_init,
                 bias_init,
                 spatial_dist = 'pois',
                 sc_dist = 'nb',
                 comm_disp = False,
                 const_disp = False,
                 name = 'dec_exp_output'):
        super(ENVIOutputLayer, self).__init__()
        
        self.input_dim = input_dim
        self.units = units
        
        self.spatial_dist = spatial_dist
        self.sc_dist = sc_dist
        self.comm_disp = comm_disp
        self.const_disp = const_disp
        
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        
        self.r = LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_r')  
        
        if(self.comm_disp):  
            
            if(self.spatial_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                                    
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init + '_d_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_d_spatial'))

            elif(self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                
            if(self.sc_dist == 'zinb'):
                self.p_sc = (self.p_spatial if (self.spatial_dist == 'zinb' or self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm') 
                            else (ConstantLayer(units, input_dim, bias_init, name = name + '_p_sc') 
                                  if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_sc')))
                                    
                self.d_sc = (self.d_spatial if (self.spatial_dist == 'zinb') 
                            else (ConstantLayer(units, input_dim, bias_init, name = name + '_d_sc') 
                                  if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_d_sc')))

            elif(self.sc_dist == 'nb' or self.sc_dist == 'full_norm'):
                
                self.p_sc = (self.p_spatial if (self.spatial_dist == 'zinb' or self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm') 
                            else (ConstantLayer(units, input_dim, bias_init, name = name + '_p_sc') 
                                  if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_r_sc')))
                
                
            if(self.spatial_dist == 'zinb' or self.sc_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                                
                self.p_sc = self.p_spatial
                
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_d_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_d_spatial'))
                
                self.d_sc = self.d_spatial

            elif(self.spatial_dist == 'nb' or self.sc_dist == 'nb' or self.spatial_dist == 'full_norm' or self.sc_dist == 'full_norm'):
                
                self.p_spatial = (ConstantLayer(units, input_dim, kernel_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                
                self.p_sc = self.p_spatial
        
        else:  
            
            if(self.spatial_dist == 'zinb'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                                    
                self.d_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_d_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_d_spatial'))

            elif(self.spatial_dist == 'nb' or self.spatial_dist == 'full_norm'):
                self.p_spatial = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_spatial')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_spatial'))
                
            if(self.sc_dist == 'zinb'):
                self.p_sc = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_sc')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_sc'))
                                    
                self.d_sc = (ConstantLayer(units, input_dim, bias_init, name = name + '_d_sc')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_d_sc'))

            elif(self.sc_dist == 'nb' or self.sc_dist == 'full_norm'):
                self.p_sc = (ConstantLayer(units, input_dim, bias_init, name = name + '_p_sc')
                          if self.const_disp else LinearLayer(units, input_dim, kernel_init, bias_init, name = name + '_p_sc'))
            
    
    
    def call(self, inputs, mode = 'spatial'):
        r = self.r(inputs)
         
        if(getattr(self, mode + '_dist') == 'zinb'):
                p = getattr(self, 'p_' + mode)(inputs)
                d = getattr(self, 'd_' + mode)(inputs)
                return(r,p,d)
                
        if(getattr(self, mode + '_dist') == 'nb' or getattr(self, mode + '_dist') == 'full_norm'):
                p = getattr(self, 'p_' + mode)(inputs)
                return(r,p)
        
        return(r)
    
    
def NormalNorm(arr):
    """
    z-scores data

    Args:
        arr (array): array to z-score
    Return:
        zscore_arr: arr z-scored
    """
    arr = np.asarray(arr)
    arr = (arr - arr.mean(axis = 0, keepdims = True))/arr.std(axis = 0, keepdims = True)
    return(arr)
    
def MatSqrtTF(Mats):
    """
    Computes psuedo matrix square root with tensorfow linear algebra on cpu

    Args:
        Mats (array): Matrices to compute square root of 
    Return:
        SqrtMats (np.array): psuedo matrix square of Mats
    """
    with tf.device('/CPU:0'):
        e,v = tf.linalg.eigh(Mats)
        e = tf.where(e < 0, 0, e)
        e = tf.math.sqrt(e)
        return(tf.linalg.matmul(tf.linalg.matmul(v, tf.linalg.diag(e)), v, transpose_b = True).numpy())


def BatchKNN(data, batch, k):
    
    kNNGraphIndex = np.zeros(shape = (data.shape[0], k))
    WeightedIndex = np.zeros(shape = (data.shape[0] ,k))
    
    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]
        
        batch_knn = sklearn.neighbors.kneighbors_graph(data[val_ind], n_neighbors=k, mode='distance', n_jobs=-1).tocoo()
        batch_knn_ind = np.reshape(np.asarray(batch_knn.col), [data[val_ind].shape[0], k])
        
        batch_knn_weight = scipy.special.softmax(-np.reshape(batch_knn.data, [data[val_ind].shape[0], k]), axis = -1)
        
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]
        WeightedIndex[val_ind] = batch_knn_weight
    return(kNNGraphIndex.astype('int'), WeightedIndex)
    
    
def GetNeighExp(spatial_data, kNN, spatial_key = 'spatial', batch_key = -1, data_key = None):
    
    """
    Computing Niche mean expression based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbours to define niche
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        data_key (str): obsm key to compute niche mean across (defualt None, uses gene expression .X)

    Return:
        NeighExp: Average geene expression in niche 
        kNNGraphIndex: indices of nearest spatial neighbours per cell
    """
    
    if(data_key is None):
        Data = spatial_data.X
    else:
        Data = spatial_data.obsm[data_key]
        
    if(batch_key == -1):        
        kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=kNN, mode='distance', n_jobs=-1).tocoo()
        kNNGraph = scipy.sparse.coo_matrix((np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)), shape=kNNGraph.shape)
        kNNGraphIndex = np.reshape(np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN])
    else:
        kNNGraphIndex, _ = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)
    
    
    return(Data[kNNGraphIndex[np.arange(spatial_data.obsm[spatial_key].shape[0])]])



def GetCOVET(spatial_data, kNN, spatial_key = 'spatial', batch_key = -1, MeanExp = None, weighted = False, cov_pc = 1):
    
        
    """
    Wrapper to compute niche covariance based on cell expression and location

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        kNN (int): number of nearest neighbours to define niche
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default -1)
        MeanExp (np.array): expression vector to shift niche covariance with
        weighted (bool): if True, weights covariance by spatial distance
    Return:
        COVET: niche covariance matrices
        kNNGraphIndex: indices of nearest spatial neighbours per cell
    """
    ExpData = spatial_data[:, spatial_data.var.highly_variable].X
    
    if(cov_pc > 0):
        ExpData = np.log(ExpData + cov_pc)
    
    if(batch_key == -1 or batch_key not in spatial_data.obs.columns):        
        kNNGraph = sklearn.neighbors.kneighbors_graph(spatial_data.obsm[spatial_key], n_neighbors=kNN, mode='distance', n_jobs=-1).tocoo()
        kNNGraph = scipy.sparse.coo_matrix((np.ones_like(kNNGraph.data), (kNNGraph.row, kNNGraph.col)), shape=kNNGraph.shape)
        kNNGraphIndex = np.reshape(np.asarray(kNNGraph.col), [spatial_data.obsm[spatial_key].shape[0], kNN])
        WeightedIndex = scipy.special.softmax(-np.reshape(kNNGraph.data, [spatial_data.obsm[spatial_key].shape[0], kNN]), axis = -1)
    else:
        kNNGraphIndex, WeightedIndex = BatchKNN(spatial_data.obsm[spatial_key], spatial_data.obs[batch_key], kNN)
        
    if not weighted:
        WeightedIndex = np.ones_like(WeightedIndex)/kNN
        
    if(MeanExp is None):
        DistanceMatWeighted = (ExpData.mean(axis = 0)[None, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]) * np.sqrt(WeightedIndex)[:, :, None] * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis= -1)))[:, None, None]
    else:
        DistanceMatWeighted = (MeanExp[:, None, :] - ExpData[kNNGraphIndex[np.arange(ExpData.shape[0])]]) * np.sqrt(WeightedIndex)[:, :, None] * np.sqrt(1 / (1 - np.sum(np.square(WeightedIndex), axis= -1)))[:, None, None]

    COVET = np.matmul(DistanceMatWeighted.transpose([0,2,1]), DistanceMatWeighted)
    COVET = COVET + COVET.mean() * 0.00001 * np.expand_dims(np.identity(COVET.shape[-1]), axis=0) 
    return(COVET, kNNGraphIndex)

def GetCov(spatial_data, k, g, genes, cov_dist, spatial_key = 'spatial', batch_key = -1, cov_pc = 1):
    
    """
    Compte niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbours to define niche
        g (int): number of HVG to compute niche covariance matricies
        genes (list of str): list of genes to keep for niche covariance
        cov_dist (str): distribution to transform niche covariance matrices to fit into
        batch_key (str): obs key for batch informationm (default -1, for no batch)

    Return:
        COVET: raw, untransformed niche covariance matrices
        COVET_SQRT: covariance matrices transformed into chosen cov_dist
        NeighExp: Average geene expression in niche 
        CovGenes: Genes used for niche covariance 
    """
        
        
    spatial_data.layers['log'] = np.log(spatial_data.X + 1)

    if(g == -1):
        CovGeneSet = np.arange(spatial_data.shape[-1])
        spatial_data.var.highly_variable = True
    else:
        sc.pp.highly_variable_genes(spatial_data, n_top_genes = g, layer = 'log')
        if(g == 0):
            spatial_data.var.highly_variable = False
        if(len(genes) > 0):
            spatial_data.var['highly_variable'][genes] = True
    
    CovGeneSet = np.where(np.asarray(spatial_data.var.highly_variable))[0]
    CovGenes = spatial_data.var_names[CovGeneSet]
    
    COVET, kNNGraphIndex = GetCOVET(spatial_data, k, spatial_key = spatial_key, batch_key = batch_key, weighted = False, cov_pc = cov_pc)
    NicheMat = spatial_data.X[kNNGraphIndex[np.arange(spatial_data.shape[0])]]
    
    if(cov_dist == 'norm'):
        COVET_SQRT = COVET.reshape([COVET.shape[0], -1])
        COVET_SQRT = (COVET_SQRT - COVET_SQRT.mean(axis = 0, keepdims = True)) / COVET_SQRT.std(axis = 0, keepdims = True)
    if(cov_dist == 'OT'):
        COVET_SQRT = MatSqrtTF(COVET)
    else:
        COVET_SQRT = np.copy(COVET)
    
    return(COVET.astype('float32'), COVET_SQRT.astype('float32'), NicheMat.astype('float32'), CovGenes)


def COVET(data, k=8, g=64,  genes=[], spatial_key = 'spatial', batch_key = -1, cov_pc = 1):
    
    """
    Compte niche covariance matrices for spatial data

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbours to define niche
        g (int): number of HVG to compute niche covariance matricies
        genes (list of str): list of genes to keep for niche covariance (if empty, just uses HVG)
        batch_key (str): obs key for batch informationm (default -1, for no batch)
        cov_pc (float): log psuedo-count for COVET computation (if 0, use unlogged values)

    Return:
        COVET: raw, untransformed niche covariance matrices
        COVET_SQRT: covariance matrices transformed into chosen cov_dist
        NeighExp: Average geene expression in niche 
        CovGenes: Genes used for niche covariance 
    """
      
    
    COVET, COVET_SQRT, _, CovGenes = GetCov(data, k, g, genes, 'OT', spatial_key = spatial_key, batch_key = batch_key, cov_pc = cov_pc)

    data.obsm['COVET'] = COVET.astype('float32')
    data.obsm['COVET_SQRT'] = COVET_SQRT.astype('float32')
    data.uns['COVET_Genes'] = CovGenes

    return(data)

class ENVI():
    
    """
    ENVI Integrates spatial and single-cell data
    
    Parameters: 
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial' indicating spatial location of spot/segmented cell
        sc_data (anndata): anndata with sinlge cell data
        spatial_key (str): obsm key name with physical location of spots/cells (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data (default 'batch' if exists on .obs, set -1 to ignore)
        num_layers (int): number of neural network for decoders and encoders (default 3)
        num_neurons (int): number of neurons in each layer (default 1024)
        latent_dim (int): size of ENVI latent dimention (size 512)
        k_nearest (int): number of physical neighbours to describe niche (default 8)
        num_cov_genes (int): number of HVGs to compute niche covariance with default (64), if -1 takes all genes
        cov_genes (list of str): manual genes to compute niche with (default [])
        num_HVG (int): number of HVGs to keep for sinlge cell data (default 2048), if -1 takes all genes
        sc_genes (list of str): manual genes to keep for sinlge cell data (default [])
        spatial_dist (str): distribution used to describe spatial data (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm') 
        sc_dist (str): distribution used to describe sinlge cell data (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        cov_dist (str): distribution used to describe niche covariance from spatial data (default OT, could be 'OT', 'wish' or 'norm')
        prior_dist (str): prior distribution for latent (default normal)
        comm_disp (bool): if True, spatial_dist and sc_dist share dispersion parameter(s) (default False)
        const_disp (bool): if True, dispertion parameter(s) are only per gene, rather there per gene per sample (default False)
        spatial_coeff (float): coefficient for spatial expression loss in total ELBO (default 1.0)
        sc_coeff (float): coefficient for sinlge cell expression loss in total ELBO (default 1.0)
        cov_coeff (float): coefficient for spatial niche loss in total ELBO (default 1.0)
        kl_coeff (float): coefficient for latent prior loss in total ELBO (default 1.0)
        skip (bool): if True, neural network has skip connections (default True)
        log_input (float): if larger than zero, a log is applied to input with pseudocount of log_input (default 0.0)
        cov_pc (float): if larger than zero, log is applied to spatial_data with pseudocount spatial_pc for calculation of spatial covariance (default 1.0)
        spatial_pc (float): if larger than zero, log is applied to spatial_data with pseudocount spatial_pc (default 0.0)
        sc_pc (float): if larger than zero, log is applied to spatial_data with pseudocount spatial_pc (default 0.0)
        lib_size (float or Bool) = if true, performs median library size, if number, normalize library size to it, if False does nothing (default False)
        z_score (float): if True and spatial/sc_dist are 'norm' or 'full_norm', spatial and sinlge cell data are z-scored (default False)
        agg (str or np.array): aggregation function of loss factors, 
                               'mean' will average across neurons, 
                               'sum' will sum across neurons (makes a difference because different number of genes for spatial and sinlge cell data), 
                                var will take a per-gene average weighed by elements in anndata.var[var]
        init_scale_out (float): scale for VarianceScaling normalization for output layer, (default 0.1) 
        init_scale_enc (float): scale for VarianceScaling normalization for encoding layer, (default 0.1) 
        init_scale_layer (float): scale for VarianceScaling normalization for regular layers, (default 0.1) 
        stable (float): pseudo count for Rate Parameter for Log Liklihood to stabelize training
    """ 

            
    def __init__(self,
                 spatial_data = None, 
                 sc_data = None, 
                 spatial_key = 'spatial',
                 batch_key = 'batch',
                 num_layers = 3, 
                 num_neurons = 1024, 
                 latent_dim = 512,
                 k_nearest = 8,
                 num_cov_genes = 64,
                 cov_genes = [],
                 num_HVG = 2048,
                 sc_genes = [],
                 spatial_dist = 'pois',
                 cov_dist = 'OT',
                 sc_dist = 'nb',
                 prior_dist = 'norm', 
                 comm_disp = False,
                 const_disp = False,
                 spatial_coeff = 1,
                 sc_coeff = 1,
                 cov_coeff = 1,
                 kl_coeff = 0.3, 
                 skip = True, 
                 log_input = 0.0,
                 cov_pc = 1,
                 spatial_pc = -1,
                 sc_pc = -1,
                 z_score  = False,
                 lib_size = False,
                 agg = 'mean', 
                 init_scale = 0.1, 
                 stable = 0.0,
                 save_path = None,
                 load_path = None,
                 **kwargs):
        

       
        super(ENVI, self).__init__()
        
        if(load_path is None):
           
            
            self.spatial_data = spatial_data.copy()
            self.sc_data = sc_data.copy()
            self.lib_size = lib_size

            self.num_layers = kwargs['NumLayers'] if 'NumLayers' in kwargs.keys() else num_layers 
            self.num_neurons = kwargs['NumNeurons'] if 'NumNeurons' in kwargs.keys() else num_neurons 
            self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs.keys() else latent_dim 

            self.spatial_dist = spatial_dist
            self.cov_dist = cov_dist
            self.sc_dist = sc_dist
            self.comm_disp = comm_disp
            self.const_disp = const_disp

            self.prior_dist = prior_dist

            self.spatial_coeff = spatial_coeff
            self.sc_coeff = sc_coeff
            self.cov_coeff = cov_coeff
            self.kl_coeff = kl_coeff 
            self.skip = skip
            self.agg = agg

            self.num_HVG = num_HVG
            self.sc_genes = sc_genes

            if(self.sc_data.raw is None):   
                self.sc_data.raw = self.sc_data

            if(self.spatial_data.raw is None):   
                self.spatial_data.raw = self.spatial_data


            self.overlap_genes = np.asarray(np.intersect1d(self.spatial_data.var_names, self.sc_data.var_names))
            self.spatial_data = self.spatial_data[:, list(self.overlap_genes)]

            self.sc_data.layers['log'] = np.log(self.sc_data.X + 1)
            sc.pp.highly_variable_genes(self.sc_data, n_top_genes = min(self.num_HVG, self.sc_data.shape[-1]), layer = 'log')

            if(self.num_HVG == 0):
                self.sc_data.var.highly_variable = False
            if(len(self.sc_genes) > 0):
                self.sc_data.var['highly_variable'][np.intersect1d(self.sc_genes, self.sc_data.var_names)] = True

            self.sc_data = self.sc_data[:, np.union1d(self.sc_data.var_names[self.sc_data.var.highly_variable], self.spatial_data.var_names)]

            self.non_overlap_genes = np.asarray(list(set(self.sc_data.var_names) - set(self.spatial_data.var_names)))
            self.sc_data = self.sc_data[:, list(self.overlap_genes) + list(self.non_overlap_genes)]

            if(self.lib_size):
                if(type(self.lib_size) == type(True)):
                    sc.pp.normalize_total(self.sc_data, target_sum = np.median(self.sc_data.X.sum(axis = 1)), inplace=True)
                else:
                    sc.pp.normalize_total(self.sc_data, target_sum = self.lib_size, inplace=True)

            self.k_nearest = k_nearest
            self.spatial_key = spatial_key

            if(batch_key in self.spatial_data.obs.columns):
                self.batch_key = batch_key
            else:
                self.batch_key = -1
            print("Computing COVET Matrices")


            self.num_cov_genes = min(num_cov_genes, self.spatial_data.shape[-1])
            self.cov_genes = cov_genes
            self.cov_pc = cov_pc

            self.spatial_data.obsm['COVET'], self.spatial_data.obsm['COVET_SQRT'], self.spatial_data.obsm['NicheMat'], self.CovGenes = GetCov(self.spatial_data, self.k_nearest, self.num_cov_genes, self.cov_genes, self.cov_dist, spatial_key = self.spatial_key, batch_key = self.batch_key, cov_pc = self.cov_pc)


            self.overlap_num = self.overlap_genes.shape[0]
            self.cov_gene_num = self.spatial_data.obsm['COVET_SQRT'].shape[-1]
            self.full_trans_gene_num = self.sc_data.shape[-1]




            if((self.agg != 'sum') and (self.agg != 'mean')):
                self.agg_spatial = self.spatial_data.var[self.agg].astype('float32') 
                self.agg_sc = self.sc_data.var[self.agg].astype('float32') 
            else:
                self.agg_spatial = self.agg
                self.agg_sc = self.agg

            self.spatial_pc = spatial_pc
            self.sc_pc = sc_pc
            
            self.log_spatial = False
            self.log_sc = False
            
            self.z_score = z_score
            self.log_input = log_input
            

            
            if((self.spatial_dist == 'norm' or self.spatial_dist == 'full_norm' and self.spatial_pc > 0) or (self.spatial_pc > (1 - self.spatial_data.X.min()))):
                
                self.log_spatial = True
                self.spatial_pc = self.spatial_pc
                self.spatial_data.uns['log_pc'] = self.spatial_pc
                self.spatial_data.X = np.log(self.spatial_data.X + self.spatial_data.uns['log_pc'])

            if(((self.sc_dist == 'norm' or self.sc_dist == 'full_norm' and self.sc_pc > 0) or (self.sc_pc > (1 - self.sc_data.X.min())))):
                
                self.log_sc = True
                self.sc_pc = self.sc_pc
                self.sc_data.uns['log_pc'] = self.sc_pc
                self.sc_data.X = np.log(self.sc_data.X + self.sc_data.uns['log_pc'])

            self.InitScale = np.abs(self.spatial_data.X).mean()

            if(self.z_score and (self.sc_dist == 'norm' or self.sc_dist == 'full_norm') and (self.spatial_dist == 'norm' or self.spatial_dist == 'full_norm')):
                self.spatial_data.var['mean'] = self.spatial_data.X.mean(axis = 0)
                self.spatial_data.var['std'] = self.spatial_data.X.std(axis = 0)

                self.spatial_data.X = (self.spatial_data.X - self.spatial_data.var['mean'][None, :])/self.spatial_data.var['std'][None, :]

                self.sc_data.var['mean'] = self.sc_data.X.mean(axis = 0)
                self.sc_data.var['std'] = self.sc_data.X.std(axis = 0)

                self.sc_data.X = (self.sc_data.X - self.sc_data.var['mean'][None, :])/self.sc_data.var['std'][None, :]

                self.InitScale = 1

            if(self.log_input > -min(self.sc_data.X.min(), self.sc_data.X.min())):
                self.log_input = kwargs['LogInput'] if 'LogInput' in kwargs.keys() else log_input 
            else:
                self.log_input = 0

            self.stable = stable
            self.init_scale = init_scale


            self.enc_layers = []
            self.dec_exp_layers = []
            self.dec_cov_layers = []


            self.initializer_layers = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = np.sqrt(self.init_scale/self.num_neurons)/self.InitScale)
            self.initializer_enc = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = np.sqrt(self.init_scale/self.num_neurons))
            self.initializer_output_cov = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = np.sqrt(self.init_scale/self.num_neurons))
            self.initializer_output_exp = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev = np.sqrt(self.init_scale/self.overlap_genes.shape[0]))

            print("Initializing VAE")

            for i in range(self.num_layers - 1):

                self.enc_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                            kernel_initializer = self.initializer_layers, 
                                                            bias_initializer = self.initializer_layers, 
                                                            name = 'enc_' + str(i)))

                self.dec_exp_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                               kernel_initializer = self.initializer_layers, 
                                                               bias_initializer = self.initializer_layers,
                                                               name = 'dec_exp_' + str(i)))

                self.dec_cov_layers.append(tf.keras.layers.Dense(units = self.num_neurons, 
                                                               kernel_initializer = self.initializer_layers, 
                                                               bias_initializer = self.initializer_layers,
                                                               name = 'dec_cov_' + str(i)))


            self.enc_layers.append(tf.keras.layers.Dense(units = 2 * latent_dim, 
                                                        kernel_initializer = self.initializer_enc, 
                                                        bias_initializer = self.initializer_enc,
                                                        name = 'enc_output'))

            self.dec_exp_layers.append(ENVIOutputLayer(input_dim = self.num_neurons, 
                                                     units = self.full_trans_gene_num, 
                                                     spatial_dist = self.spatial_dist, 
                                                     sc_dist = self.sc_dist, 
                                                     comm_disp = self.comm_disp, 
                                                     const_disp = self.const_disp, 
                                                     kernel_init = self.initializer_output_exp, 
                                                     bias_init = self.initializer_output_exp,
                                                     name = 'dec_exp_output'))


            self.dec_cov_layers.append(tf.keras.layers.Dense(units = int(self.cov_gene_num * (self.cov_gene_num + 1) / 2), 
                                                           kernel_initializer = self.initializer_output_cov, 
                                                           bias_initializer = self.initializer_output_cov,
                                                           name = 'dec_cov_output'))



            if not (save_path is None):
                self.save_path = save_path
            else:
                self.save_path = -1
            print("Finished Initializing ENVI")
            
        else:
            self.load_path = load_path
            
            with open(self.load_path, 'rb') as handle:   
                envi_model = pickle.load(handle)
                
            for key,val in zip(envi_model.keys(),envi_model.values()):
                setattr(self, key, val)
            
            print("Finished loading ENVI model")
            
    @tf.function
    def encode_nn(self, Input):
        
        """
        Encoder forward pass
        
        Args:
            Input (array): input to encoder NN (size of #genes in spatial data + confounder)
        Returns:
            Output (array): NN output
        """
        
        Output = Input
        for i in range(self.num_layers - 1):
            Output = self.enc_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(self.enc_layers[-1](Output))

    @tf.function
    def decode_exp_nn(self, Input):
        
        """
        Expression decoder forward pass
        
        Args:
            Input (array): input to expression decoder NN (size of latent dimension + confounder)
            
        Returns:
            Output (array): NN output
        """
        
        Output = Input
        for i in range(self.num_layers - 1):
            Output = self.dec_exp_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(Output)

    @tf.function
    def decode_cov_nn(self, Output):
        
        """
        Covariance (niche) decoder forward pass
        
        Args:
            Input (array): input to niche decoder NN (size of latent dimension + confounder)
            
        Returns:
            Output (array): NN output
        """
        

        for i in range(self.num_layers - 1):
            Output = self.dec_cov_layers[i](Output) + (Output if (i > 0 and self.skip) else 0)
            Output = tf.nn.leaky_relu(Output)
        return(self.dec_cov_layers[-1](Output))


    @tf.function
    def encode(self, x, mode = 'sc'):
        
        """
        Appends confounding variable to input and generates an encoding
        
        Args:
            x (array): input to encoder (size of #genes in spatial data)
            mode (str): 'sc' for sinlge cell, 'spatial' for spatial data 
            
        Return:
            mean (array): mean parameter for latent variable
            log_std (array): log of the standard deviation for latent variable
        """

        conf_const = 0 if mode == 'spatial' else 1 
        if(self.log_input > 0):
            x = tf.math.log(x + self.log_input)
        x_conf = tf.concat([x, tf.one_hot(conf_const * tf.ones(x.shape[0], dtype=tf.uint8), 2, dtype=tf.keras.backend.floatx())], axis = -1)
        return(tf.split(self.encode_nn(x_conf), num_or_size_splits = 2, axis = 1))

    @tf.function
    def exp_decode(self, x, mode = 'sc'):    
        """
        Appends confounding variable to latent and generates an output distribution
        
        Args:
            x (array): input to expression decoder (size of latent dimension)
            mode (str): 'sc' for sinlge cell, 'spatial' for spatial data 
            
        Return:
            Output paramterizations for chosen expression distributions
        """
        conf_const = 0 if mode == 'spatial' else 1
        x_conf = tf.concat([x, tf.one_hot(conf_const * tf.ones(x.shape[0], dtype=tf.uint8), 2, dtype=tf.keras.backend.floatx())], axis = -1)
        DecOut = self.decode_exp_nn(x_conf)
        
        if (getattr(self, mode + '_dist') == 'zinb'):
            output_r, output_p, output_d = self.dec_exp_layers[-1](DecOut, mode)
        
        
            return tf.nn.softplus(output_r) + self.stable, output_p, tf.nn.sigmoid(0.01 * output_d - 2)
        if (getattr(self, mode + '_dist') == 'nb'):
            output_r, output_p = self.dec_exp_layers[-1](DecOut, mode)
            
            return tf.nn.softplus(output_r) + self.stable, output_p
        if (getattr(self, mode + '_dist') == 'pois'):
            output_l = self.dec_exp_layers[-1](DecOut, mode)
            return tf.nn.softplus(output_l) + self.stable
        if (getattr(self, mode + '_dist') == 'full_norm'):
            output_mu, output_logstd = self.dec_exp_layers[-1](DecOut, mode)
            return output_mu, output_logstd
        if (getattr(self, mode + '_dist') == 'norm'):
            output_mu =  self.dec_exp_layers[-1](DecOut, mode)
            return output_mu
        
    @tf.function
    def cov_decode(self, x):
        """
        Generates an output distribution for niche data
        
        Args:
            x (array): input to covariance decoder (size of latent dimension)
            
        Return:
            Output paramterizations for chosen niche distributions
        """
        
        DecOut = self.decode_cov_nn(x)
        if(self.cov_dist == 'wish'):
            TriMat = tfp.math.fill_triangular(DecOut)
            TriMat = tf.linalg.set_diag(TriMat, tf.math.softplus(tf.linalg.diag_part(TriMat)))
            return(TriMat)
        elif(self.cov_dist == 'norm'):
            TriMat = tfp.math.fill_triangular(DecOut)
            return(0.5 * TriMat + 0.5 * tf.tranpose(TriMat, [0,2,1]))
        elif(self.cov_dist == 'OT'):
            TriMat = tfp.math.fill_triangular(DecOut)
            return(tf.matmul(TriMat, TriMat, transpose_b = True))
    
    @tf.function
    def enc_mean(self, mean, logstd):
        """
        Returns posterior mean given latent parametrization, which is not the mean varialbe for a log_normal prior
        
        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            Posterior mean for latent
        """
        if(self.prior_dist == 'norm'):
            return mean
        elif(self.prior_dist == 'log_norm'):
            return tf.exp(mean + tf.square(tf.exp(logstd))/2)
        
    @tf.function
    def reparameterize(self, mean, logstd):
        """
        Samples from latent using te reparameterization trick
        
        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            sample from latent
        """
        reparm = tf.random.normal(shape=mean.shape, dtype = tf.keras.backend.floatx()) * tf.exp(logstd) + mean
        if(self.prior_dist == 'norm'):
            return reparm
        elif(self.prior_dist == 'log_norm'):
            return tf.exp(reparm)


    @tf.function
    def compute_loss(self, spatial_sample, cov_sample, sc_sample):
        
        """
        Computes ENVI liklihoods
        
        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): niche covariance data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted to spatial genes
        Return:
            spatial_like: ENVI liklihood for spatial expression
            cov_like: ENVI liklihood for covariance data
            sc_like: ENVI liklihood for sinlge cell data
            kl: KL divergence between posterior latent and prior
        """
            
        mean_spatial, logstd_spatial = self.encode(spatial_sample[:, :self.overlap_num], mode = 'spatial')
        mean_sc, logstd_sc = self.encode(sc_sample[:, :self.overlap_num], mode = 'sc')
         
        z_spatial = self.reparameterize(mean_spatial, logstd_spatial)
        z_sc = self.reparameterize(mean_sc, logstd_sc)
        

            
        if (self.spatial_dist == 'zinb'):
            spatial_r, spatial_p, spatial_d = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_zinb_pdf(spatial_sample, 
                                                       spatial_r[:, :spatial_sample.shape[-1]], 
                                                       spatial_p[:, :spatial_sample.shape[-1]], 
                                                       spatial_d[:, :spatial_sample.shape[-1]], agg = self.agg_spatial), axis = 0)
        if (self.spatial_dist == 'nb'):
            spatial_r, spatial_p = self.exp_decode(z_spatial, mode = 'spatial')                                 
            spatial_like = tf.reduce_mean(log_nb_pdf(spatial_sample, 
                                                     spatial_r[:, :spatial_sample.shape[-1]],
                                                     spatial_p[:, :spatial_sample.shape[-1]], agg = self.agg_spatial), axis = 0)
        if (self.spatial_dist == 'pois'):
            spatial_l = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_pos_pdf(spatial_sample, 
                                                      spatial_l[:, :spatial_sample.shape[-1]], agg = self.agg_spatial), axis = 0)
        if (self.spatial_dist == 'full_norm'):
            spatial_mu, spatial_logstd = self.exp_decode(z_spatial, mode = 'spatial')                                         
            spatial_like = tf.reduce_mean(log_normal_pdf(spatial_sample, 
                                                         spatial_mu[:, :spatial_sample.shape[-1]], 
                                                         spatial_logstd[:, :spatial_sample.shape[-1]], agg = self.agg_spatial), axis = 0)
        if (self.spatial_dist == 'norm'):
            spatial_mu = self.exp_decode(z_spatial, mode = 'spatial')
            spatial_like = tf.reduce_mean(log_normal_pdf(spatial_sample, 
                                                         spatial_mu[:, :spatial_sample.shape[-1]], 
                                                         tf.zeros_like(spatial_sample), agg = self.agg_spatial), axis = 0)

        
        if (self.sc_dist == 'zinb'):
            sc_r, sc_p, sc_d = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_zinb_pdf(sc_sample, 
                                                  sc_r, 
                                                  sc_p, 
                                                  sc_d, agg = self.agg_sc), axis = 0) 
        if (self.sc_dist == 'nb'):
            sc_r, sc_p = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_nb_pdf(sc_sample, 
                                                  sc_r, 
                                                  sc_p, agg = self.agg_sc), axis = 0)
        if (self.sc_dist == 'pois'):
            sc_l = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_pos_pdf(sc_sample, 
                                              sc_l, agg = self.agg_sc), axis = 0)
        if (self.sc_dist == 'full_norm'):
            sc_mu, sc_std = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_normal_pdf(sc_sample, 
                                                      sc_mu, 
                                                      sc_std, agg = self.agg_sc), axis = 0)
        if (self.sc_dist == 'norm'):
            sc_mu = self.exp_decode(z_sc, mode = 'sc')
            sc_like = tf.reduce_mean(log_normal_pdf(sc_sample, 
                                                    sc_mu, 
                                                    tf.zeros_like(sc_sample), agg = self.agg_sc), axis = 0)
        
        
        if(self.cov_dist == 'wish'):
            cov_mu = self.cov_decode(z_spatial)
            cov_like = tf.reduce_mean(log_wish_pdf(cov_sample, 
                                                   cov_mu, agg = self.agg), axis = 0)
        elif(self.cov_dist == 'norm'):
            cov_mu = tf.reshape(self.cov_decode(z_spatial), [spatial_sample.shape[0], -1])
            cov_like = tf.reduce_mean(log_normal_pdf(tf.reshape(cov_sample, [cov_sample.shape[0], -1]), 
                                                     cov_mu, 
                                                     tf.zeros_like(cov_mu), agg = self.agg), axis = 0)
        elif(self.cov_dist == 'OT'):
            cov_mu = self.cov_decode(z_spatial)
            cov_like = tf.reduce_mean(OTDistance(cov_sample, 
                                                 cov_mu, agg = self.agg), axis = 0)
    
    
        if(self.prior_dist == 'norm'):
                kl_spatial = tf.reduce_mean(NormalKL(mean_spatial, logstd_spatial, agg = self.agg), axis = 0)
                kl_sc = tf.reduce_mean(NormalKL(mean_sc, logstd_sc, agg = self.agg), axis = 0)
        elif(self.prior_dist == 'log_norm'):
                kl_spatial = tf.reduce_mean(LogNormalKL(logstd_spatial, logstd, agg = self.agg), axis = 0)
                kl_sc = tf.reduce_mean(LogNormalKL(mean_sc, logstd_sc, agg = self.agg), axis = 0)
                
        kl = 0.5 * kl_spatial + 0.5 * kl_sc
    

        return(spatial_like, cov_like, sc_like, kl)
 


    def GetCovMean(self, cov_mat):
        """
        Reconstructs true covarianace (untransformed)
        
        Args:
            cov_mat (array/tensor): transformed covariance matricies to untransform
        Return:
            untransform covariance matrices
        """
        if(self.cov_dist == 'wish'):
            return(cov_mat * tf.sqrt(cov_mat.shape[-1]))
        elif(self.cov_dist == 'OT'):
            return(tf.mamtul(cov_mat, cov_mat))
        else:
            return(cov_mat)
    
    def GetMeanSample(self, decode, mode = 'spatial'):
        """
        Computes mean of expression distribution 
        
        Args:
            decode (list or array/tensor): parameter of distribution 
            mode (str): modality of data to compute distribution mean of (default 'spatial')
        Return:
            distribution mean from parameterization
        """
        if (getattr(self, mode + '_dist') == 'zinb'):
            return(decode[0] * tf.exp(decode[1]) * (1 - decode[2]))
        elif (getattr(self, mode + '_dist') == 'nb'):
            #return(decode[0])
            return(decode[0] * tf.exp(decode[1]))
        elif (getattr(self, mode + '_dist') == 'pois'):
            return(decode)
        elif (getattr(self, mode + '_dist') == 'full_norm'):
            return(decode[0])
        elif (getattr(self, mode + '_dist') == 'norm'):
            return(decode)
    
    def cluster_rep(self):
        import phenograph
        comm_emb = phenograph.cluster(np.concatenate((self.spatial_data.obsm['envi_latent'], self.sc_data.obsm['envi_latent']), axis = 0))[0]
        
        self.spatial_data.obs['latent_cluster'] = comm_emb[:self.spatial_data.shape[0]]
        self.sc_data.obs['latent_cluster'] = comm_emb[self.spatial_data.shape[0]:]
    
    def latent_rep(self, NumDiv = 16, data = None, mode = None): 
        """
        Compute latent embeddings for spatial and single cell data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
        Return:
            no return, adds 'envi_latent' ENVI.spatial_data.obsm and ENVI.spatial_data.obsm
        """
        
        if(data is None):
            self.spatial_data.obsm['envi_latent'] = np.concatenate([self.encode(np.array_split(self.spatial_data.X.astype(tf.keras.backend.floatx()), 
                                                                   NumDiv, axis = 0)[_], mode = 'spatial')[0].numpy() for _ in range(NumDiv)], axis = 0)


            self.sc_data.obsm['envi_latent'] = np.concatenate([self.encode(np.array_split(self.sc_data[:, self.spatial_data.var_names].X.astype(tf.keras.backend.floatx()), 
                                                                  NumDiv, axis = 0)[_], mode = 'sc')[0].numpy() for _ in range(NumDiv)], axis = 0)
        
        else:
            
            data = data.copy()
            
            
            if(mode == 'spatial'):
                
                if(not set(self.spatial_data.var_names).issubset(set(data.var_names))):
                    print("(Spatial) Data does not contain trained gene")
                    return(-1)
                
                data = data[:, self.spatial_data.var_names]
                
                if(self.log_spatial):
                    data.X = np.log(data.X + self.spatial_data.uns['log_pc'])
                    
                if(self.z_score):
                    data.X  = (data.X - self.spatial_data.var['mean'])/self.spatial_data.var['std'] 
            
            else:
                
                if(not set(self.spatial_data.var_names).issubset(set(data.var_names))):
                    print("(sc) Data does not contain trained gene")
                    return(-1)
                
                data = data[:, self.sc_data.var_names]
                
                if(self.log_spatial):
                    data.X = np.log(data.X + self.sc_data.uns['log_pc'])
                    
                if(self.z_score):
                    data.X  = (data.X - self.sc_data.var['mean'])/self.sc_data.var['std'] 
                    
                    
            envi_latent = np.concatenate([self.encode(np.array_split(data[:, self.spatial_data.var_names].X.astype(tf.keras.backend.floatx()),
                                                                   NumDiv, axis = 0)[_], mode = mode)[0].numpy() for _ in range(NumDiv)], axis = 0)
            return(envi_latent)

    def pred_type(self, pred_on = 'sc', key_name = 'cell_type', ClassificationModel = sklearn.neural_network.MLPClassifier(alpha=0.01, max_iter = 100, verbose = False)):
        """
        Transfer labeling from one modality to the other using latent embeddings
        
        Args:
            pred_on (str): what modality to predict labeling for (default 'sc', i.e. transfer from spatial_data to single cell data)
            key_name (str): obsm key name for labeling (default 'cell_type')
            ClassificationModel (sklearn model): Classification model to learn cell labelings (defualt sklearn.neural_network.MLPClassifier)
        Return:
            no return, adds key_name with cell labelings to ENVI.spatial_data.obsm or ENVI.spatial_data.obsm, depending on pred_on
        """
                    
        if(pred_on == 'sc'):
            ClassificationModel.fit(self.spatial_data.obsm['envi_latent'], self.spatial_data.obs[key_name]) 
            self.sc_data.obs[key_name + '_envi'] = ClassificationModel.predict(self.sc_data.obsm['envi_latent']) 
            self.spatial_data.obs[key_name + '_envi'] = self.spatial_data.obs[key_name]
            print("Finished Transfering labels to single cell data! See " +  key_name +"_envi in obsm of ENVI.sc_data")
        else:
            ClassificationModel.fit(self.sc_data.obsm['envi_latent'], self.sc_data.obs[key_name])
            self.spatial_data.obs[key_name + '_envi'] = ClassificationModel.predict(self.spatial_data.obsm['envi_latent'])  
            self.sc_data.obs[key_name + '_envi'] = self.sc_data.obs[key_name]
            print("Finished Transfering labels to spatial data! See " +  key_name +"_envi in obsm of ENVI.spatial_data")

    
    def impute(self, NumDiv = 16, return_raw = True, data = None):
        """
        Imput full transcriptome for spatial data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
            return_raw (bool): if True, un-logs and un-zcores imputation if either were chosen
        Return:
            no return, adds 'imputation' to ENVI.spatial_data.obsm
        """
        
        if(data is None):
            decode = np.concatenate([self.GetMeanSample(self.exp_decode(np.array_split(self.spatial_data.obsm['envi_latent'], NumDiv, axis = 0)[_], mode = 'sc'), mode = 'sc').numpy() for _ in range(NumDiv)], axis = 0)

            imputation = pd.DataFrame(decode, columns = self.sc_data.var_names, index = self.spatial_data.obs_names)

            if(return_raw):
                if(self.z_score):
                    imputation = imputation * self.sc_data.var['std'] + self.sc_data.var['mean']
                if(self.log_sc):
                    imputation = np.exp(imputation) - self.sc_data.uns['log_pc']
                    imputation[imputation < 0] = 0

            self.spatial_data.obsm['imputation'] = imputation
        else:
            
            latent = self.latent_rep(data = data, mode = 'spatial')
            
            decode = np.concatenate([self.GetMeanSample(self.exp_decode(np.array_split(latent, NumDiv, axis = 0)[_], mode = 'sc'), mode = 'sc').numpy() for _ in range(NumDiv)], axis = 0)

            imputation = pd.DataFrame(decode, columns = self.sc_data.var_names, index = self.spatial_data.obs_names)

            if(return_raw):
                if(self.z_score):
                    imputation = imputation * self.sc_data.var['std'] + self.sc_data.var['mean']
                if(self.log_sc):
                    imputation = np.exp(imputation) - self.sc_data.uns['log_pc']
                    imputation[imputation < 0] = 0

            return(imputation)
        
        print("Finished imputing missing gene for spatial data! See 'imputation' in obsm of ENVI.spatial_data")
     
    def infer_COVET(self, NumDiv = 16, data = None):
        """
        Infer covariance niche composition for single cell data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
            revert (bool): if True, computes actual covariance, if False, computes transformed covariance (default False)
        Return:
            no return, adds 'COVET_SQRT' or 'COVET' to ENVI.sc_data.obsm
        """
        
        if(data is None):
            self.sc_data.obsm['COVET_SQRT'] = np.concatenate([self.cov_decode(np.array_split(self.sc_data.obsm['envi_latent'], NumDiv, axis = 0)[_]) 
                                     for _ in range(NumDiv)], axis = 0)
            self.sc_data.obsm['COVET'] = np.concatenate([np.linalg.matrix_power(np.array_split(self.sc_data.obsm['COVET_SQRT'], NumDiv, axis = 0)[_], 2) 
                             for _ in range(NumDiv)], axis = 0)
        else:
            latent = self.latent_rep(data = data, mode = 'sc')
            COVET_SQRT = np.concatenate([self.cov_decode(np.array_split(latent, NumDiv, axis = 0)[_]) 
                                     for _ in range(NumDiv)], axis = 0)
            COVET = np.concatenate([np.linalg.matrix_power(np.array_split(COVET_SQRT, NumDiv, axis = 0)[_], 2) 
                             for _ in range(NumDiv)], axis = 0)
            return(COVET_SQRT, COVET)

    def infer_COVET_spatial(self, NumDiv = 16):
        """
        Reconstruct covariance niche composition for spatial data
        
        Args:
            NumDiv (int): number of splits for forward pass to allow to fit in gpu
  
        Return:
            no return, adds 'COVET_SQRT' or 'COVET' to ENVI.sc_data.obsm
        """
            
        self.spatial_data.obsm['COVET_SQRT_envi'] = np.concatenate([self.cov_decode(np.array_split(self.spatial_data.obsm['envi_latent'], NumDiv, axis = 0)[_]) 
                                 for _ in range(NumDiv)], axis = 0)
        self.spatial_data.obsm['COVET_envi'] = np.concatenate([np.linalg.matrix_power(np.array_split(self.spatial_data.obsm['COVET_SQRT_envi'], NumDiv, axis = 0)[_], 2) 
                         for _ in range(NumDiv)], axis = 0)

    
    def reconstruct_niche(self, k = 8, niche_key = 'cell_type', pred_key = None, norm_reg = False, cluster = False, res = 0.5, data = None):
        
        """
        Infer niche composition for single cell data
        
        Args:
            k (float): k for kNN regression on covariance matrices (defulat 32)
            niche_key (str): spaital obsm key to reconstruct niche from (default 'cell_type')
            pred_key (str): spatial & single cell obsm key to split up kNN regression by (default None)
            gpu (bool): if True, uses gpu for kNN regression (default False)
            norm_reg (bool): if True, cell type enrichement in normalized by the number of cells per type (default False)
            cluster (bool): if True, clusters covariance data and produces niche based on average across cluster, k is parameter for phenograph (default False)
            res (float): resolution parameter for leiden clustering in phenograph (default 0.5)

            no return, adds 'niche_by_type' to ENVI.sc_data.obsm
        """
            
#        print(cluster)    

        
        import sklearn.preprocessing 
        
        LabelEnc = sklearn.preprocessing.LabelBinarizer().fit(self.spatial_data.obs[niche_key])
        spatialCellTypeEncoding = LabelEnc.transform(self.spatial_data.obs[niche_key])
        self.spatial_data.obsm[niche_key + '_enc'] = spatialCellTypeEncoding
        CellTypeName = LabelEnc.classes_
          
        NeighCellType = GetNeighExp(self.spatial_data, self.k_nearest, 
                                    data_key = (niche_key + '_enc'), spatial_key = self.spatial_key, batch_key = self.batch_key)
        
        NeighCellType = NeighCellType.sum(axis = 1).astype('float32')
        
        self.spatial_data.obsm['niche_by_type'] = pd.DataFrame(NeighCellType, columns = CellTypeName, index = self.spatial_data.obs_names)
    
        

        import os, sys
        
        if(cluster):
            import phenograph

        if(data is None):
            self.infer_COVET(16, False)

            if(pred_key is None):
                CovFit = self.spatial_data.obsm['COVET_SQRT']
                CovPred = self.sc_data.obsm['COVET_SQRT']

                NeighFit = self.spatial_data.obsm['niche_by_type']

                if(norm_reg):
                    NeighFit = NeighFit/self.spatial_data.obsm['cell_type_enc'].sum(axis = 0, keepdims = True)
                    NeighFit = NeighFit/NeighFit.sum(axis = 1, keepdims = True) * self.k_nearest


                if(cluster):
                    with HiddenPrints():
                        phenoclusters = phenograph.cluster(np.concatenate((CovFit.reshape([CovFit.shape[0], -1]) , CovPred.reshape([CovPred.shape[0], -1])), axis = 0), clustering_algo = 'leiden',
                                                               k = k, resolution_parameter = res)

                    phenoclusters_fit = phenoclusters[:CovFit.shape[0]]
                    phenoclusters_pred = phenoclusters[CovFit.shape[0]:]

                    avg_niche = np.asarray([NeighFit[phenoclusters_fit == clust].mean(axis = 0) for clust in np.arange(phenoclusters.max() + 1)])
                    pred_niche = avg_niche[phenoclusters_pred]

                    self.sc_data.obsm['niche_by_type'] = pd.DataFrame(pred_niche, columns = CellTypeName, index = self.sc_data.obs_names)     
                else:
                    import sklearn.neighbors
                    regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = min(k, CovFit.shape[0]))


                    regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
                    NeighPred = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))
                    self.sc_data.obsm['niche_by_type'] = pd.DataFrame(NeighPred, columns = CellTypeName, index = self.sc_data.obs_names)
            else:

                NeighPred = np.zeros(shape = (self.sc_data.shape[0], NeighCellType.shape[-1]))
                for val in np.unique(self.sc_data.obs[pred_key]):
                    CovFit = self.spatial_data.obsm['COVET_SQRT'][self.spatial_data.obs[pred_key] == val]
                    CovPred = self.sc_data.obsm['COVET_SQRT'][self.sc_data.obs[pred_key] == val]

                    NeighFit = np.asarray(self.spatial_data.obsm['niche_by_type'][self.spatial_data.obs[pred_key] == val])

                    if(norm_reg):
                        NeighFit = NeighFit/self.spatial_data.obsm['cell_type_enc'].sum(axis = 0, keepdims = True)
                        NeighFit = NeighFit/NeighFit.sum(axis = 1, keepdims = True) * self.k_nearest

                    if(cluster):
                        with HiddenPrints():
                            phenoclusters = phenograph.cluster(np.concatenate((CovFit.reshape([CovFit.shape[0], -1]) , CovPred.reshape([CovPred.shape[0], -1])), axis = 0), clustering_algo = 'leiden',
                                                               k = k, resolution_parameter = res)[0]

                        phenoclusters_fit = phenoclusters[:CovFit.shape[0]]
                        phenoclusters_pred = phenoclusters[CovFit.shape[0]:]

                        avg_niche = np.asarray([NeighFit[phenoclusters_fit == clust].mean(axis = 0) for clust in np.arange(phenoclusters.max() + 1)])
                        NeighPred[self.sc_data.obs[pred_key] == val] = avg_niche[phenoclusters_pred]

                    else:
                        import sklearn.neighbors
                        regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = min(k, CovFit.shape[0]))


                        regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
                        NeighPred[self.sc_data.obs[pred_key] == val] = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))

                self.sc_data.obsm['niche_by_type'] = pd.DataFrame(NeighPred, columns = CellTypeName, index = self.sc_data.obs_names)
        else:
            
            sc_cov_mats = self.infer_COVET(16, False, data)

            if(pred_key is None):
                CovFit = self.spatial_data.obsm['COVET_SQRT']
                CovPred = sc_cov_mats

                NeighFit = self.spatial_data.obsm['niche_by_type']

                if(norm_reg):
                    NeighFit = NeighFit/self.spatial_data.obsm['cell_type_enc'].sum(axis = 0, keepdims = True)
                    NeighFit = NeighFit/NeighFit.sum(axis = 1, keepdims = True) * self.k_nearest


                if(cluster):
                    with HiddenPrints():
                        phenoclusters = phenograph.cluster(np.concatenate((CovFit.reshape([CovFit.shape[0], -1]) , CovPred.reshape([CovPred.shape[0], -1])), axis = 0), clustering_algo = 'leiden',
                                                               k = k, resolution_parameter = res)

                    phenoclusters_fit = phenoclusters[:CovFit.shape[0]]
                    phenoclusters_pred = phenoclusters[CovFit.shape[0]:]

                    avg_niche = np.asarray([NeighFit[phenoclusters_fit == clust].mean(axis = 0) for clust in np.arange(phenoclusters.max() + 1)])
                    pred_niche = avg_niche[phenoclusters_pred]

                    niche_by_type = pd.DataFrame(pred_niche, columns = CellTypeName, index = self.sc_data.obs_names)     
                else:
                    import sklearn.neighbors
                    regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = min(k, CovFit.shape[0]))


                    regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
                    NeighPred = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))
                    niche_by_type = pd.DataFrame(NeighPred, columns = CellTypeName, index = data.obs_names)
            else:

                NeighPred = np.zeros(shape = (data.shape[0], NeighCellType.shape[-1]))
                for val in np.unique(self.sc_data.obs[pred_key]):
                    CovFit = self.spatial_data.obsm['COVET_SQRT'][self.spatial_data.obs[pred_key] == val]
                    CovPred = sc_cov_mats[data.obs[pred_key] == val]

                    NeighFit = np.asarray(self.spatial_data.obsm['niche_by_type'][self.spatial_data.obs[pred_key] == val])

                    if(norm_reg):
                        NeighFit = NeighFit/self.spatial_data.obsm['cell_type_enc'].sum(axis = 0, keepdims = True)
                        NeighFit = NeighFit/NeighFit.sum(axis = 1, keepdims = True) * self.k_nearest

                    if(cluster):
                        with HiddenPrints():
                            phenoclusters = phenograph.cluster(np.concatenate((CovFit.reshape([CovFit.shape[0], -1]) , CovPred.reshape([CovPred.shape[0], -1])), axis = 0), clustering_algo = 'leiden',
                                                               k = k, resolution_parameter = res)[0]

                        phenoclusters_fit = phenoclusters[:CovFit.shape[0]]
                        phenoclusters_pred = phenoclusters[CovFit.shape[0]:]

                        avg_niche = np.asarray([NeighFit[phenoclusters_fit == clust].mean(axis = 0) for clust in np.arange(phenoclusters.max() + 1)])
                        NeighPred[data.obs[pred_key] == val] = avg_niche[phenoclusters_pred]

                    else:

                        import sklearn.neighbors
                        regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = min(k, CovFit.shape[0]))


                        regressor.fit(CovFit.reshape([CovFit.shape[0], -1]), NeighFit)
                        NeighPred[data.obs[pred_key] == val] = regressor.predict(CovPred.reshape([CovPred.shape[0], -1]))

                niche_by_type = pd.DataFrame(NeighPred, columns = CellTypeName, index = data.obs_names)
            
            return(niche_by_type)
#         print("Finished Niche Reconstruction! See 'niche_by_type' in obsm of ENVI.sc_data")
    
    @tf.function
    def compute_apply_gradients(self, spatial_sample, cov_sample, sc_sample):
                
        """
        Applies gradient descent step given training batch
        
        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): niche covariance data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted to spatial genes
        Return:
            spatial_like: ENVI liklihood for spatial expression
            cov_like: ENVI liklihood for covariance data
            sc_like: ENVI liklihood for sinlge cell data
            kl: KL divergence between posterior latent and prior
            nan: True if any factor in loss was nan and doesn't apply gradients
        """
            
        with tf.GradientTape() as tape:
            spatial_like,  cov_like, sc_like, kl = self.compute_loss(spatial_sample, cov_sample, sc_sample)
            loss = - self.spatial_coeff * spatial_like - self.sc_coeff * sc_like - self.cov_coeff * cov_like + 2 * self.kl_coeff * kl
        
        if(not hasattr(ENVI, 'trainable_variables')):
            self.trainable_variables = []     
            for ind, var in enumerate(self.enc_layers + self.dec_exp_layers + self.dec_cov_layers):
                self.trainable_variables = self.trainable_variables + var.weights

        gradients = tape.gradient(loss, self.trainable_variables)
        nan = False
        
#         for grad in gradients:
#             if(tf.reduce_sum(tf.cast(tf.math.is_nan(grad), tf.int8)) > 0):
#                 nan = True
        if(tf.math.is_nan(loss)):
            nan = True
            
        if(nan):
            return(spatial_like,  cov_like, sc_like, kl, True)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return (spatial_like,  cov_like, sc_like, kl, False)

    def Train(self, LR = 0.0001, batch_size = 512, epochs = np.power(2,14), verbose = 64, LR_Sched = True):
                        
        """
        ENVI training loop and computing of latent representation at end
        
        Args:
            LR (float): learning rate for training (default 0.0001)
            batch_size (int): total batch size for traning, to sample from single cell and spatial data (default 512)
            epochs (int): number of training steps (default 16384)
            split (float): train/test split of data (default 1.0)
            verbose (int): how many training step between each print statement, if -1 than no printing (default 128)
            LR_Sched (bool): if True, decreases LR by factor of 10 after 0.75 * epochs steps (default True)
        Return:
            no return, trains ENVI and adds 'envi_latent' to obsm of ENVI.sc_data and ENVI.spatial_data
        """
        
        
        print("Training ENVI for {} steps".format(epochs))

        self.LR = LR
        self.optimizer = tf.keras.optimizers.Adam(self.LR)

        
        
        spatial_data_train = self.spatial_data.X.astype(tf.keras.backend.floatx())
        cov_data_train = self.spatial_data.obsm['COVET_SQRT'].astype(tf.keras.backend.floatx())
        sc_data_train = self.sc_data.X.astype(tf.keras.backend.floatx())
        

        ## Run Dummy:
        log_pos_pdf(tf.ones([5], tf.float32), tf.ones([5], tf.float32))
        
        start_time = time.time()
        
        
        from tqdm import trange
        
        
            
        tq = trange(epochs, leave=True, desc = "")
        for epoch in tq: #range(0, epochs + 1):
            

            if((epoch == int(epochs * 0.75)) and LR_Sched):
                self.LR = LR * 0.1
                self.optimizer.lr.assign(self.LR)

            self.batch_spatial = np.random.choice(np.arange(spatial_data_train.shape[0]),
                                        min(batch_size, spatial_data_train.shape[0]), replace = False)
            
            self.batch_sc = np.random.choice(np.arange(sc_data_train.shape[0]),
                                   min(batch_size, sc_data_train.shape[0]), replace = False)

            if ((epoch % verbose == 0) and (verbose > 0)):

                
                end_time = time.time()

                loss_spatial, loss_cov, loss_sc, loss_kl = self.compute_loss(spatial_data_train[self.batch_spatial], 
                                                                             cov_data_train[self.batch_spatial], 
                                                                             sc_data_train[self.batch_sc])
                
                print_statement = 'Trn: spatial Loss: {:.5f}, SC Loss: {:.5f}, Cov Loss: {:.5f}, KL Loss: {:.5f}'.format(loss_spatial.numpy(), 
                                                                              loss_sc.numpy(), 
                                                                              loss_cov.numpy(), 
                                                                              loss_kl.numpy())
                tq.set_description(print_statement, refresh=True)
     


                start_time = time.time()

            loss_spatial,  loss_cov, loss_sc, loss_kl, nan = self.compute_apply_gradients(spatial_data_train[self.batch_spatial], 
                                         cov_data_train[self.batch_spatial], 
                                         sc_data_train[self.batch_sc])
            

            
        print("Finished Training ENVI! - calculating latent embedding, see 'envi_latent' obsm of ENVI.sc_data and ENVI.spatial_data")
    
        self.latent_rep()
        self.save_model()

            
            
    def save_model(self):
        
        if(self.save_path != -1):
            attribute_name_list  = ['spatial_data', 'sc_data', 'spatial_key', 'batch_key', 'num_layers', 'num_neurons', 'latent_dim', 'k_nearest', 'num_cov_genes',
                                    'overlap_num', 'cov_gene_num', 'full_trans_gene_num', 'log_spatial', 'log_sc', 
                               'cov_genes', 'num_HVG','sc_genes', 'spatial_dist', 'cov_dist','sc_dist','prior_dist','comm_disp', 'const_disp', 'spatial_coeff',
                               'sc_coeff', 'kl_coeff', 'skip', 'log_input', 'cov_pc', 'spatial_pc', 'sc_pc', 'z_score', 'lib_size', 'agg', 'init_scale', 'stable', 
                              'enc_layers', 'dec_exp_layers', 'dec_cov_layers']

            attribute_list = {attr:getattr(self, attr) for attr in attribute_name_list}

            directory = self.save_path
            directory = '/'.join(directory.split('/')[:-1])

            import os
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(self.save_path, 'wb') as handle:
                pickle.dump(attribute_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



@tf.function
def LogNormalKL(mean, log_std, agg = None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if(agg is None):
        return(KL)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(KL, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(KL, axis = -1))
    return(tf.reduce_mean(KL, axis = -1))
     
    
@tf.function        
def NormalKL(mean, log_std, agg = None):
    KL = 0.5 * (tf.square(mean) + tf.square(tf.exp(log_std)) - 2 * log_std)
    if(agg is None):
        return(KL)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(KL, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(KL, axis = -1))
    
    return(tf.reduce_mean(KL, axis = -1))

@tf.function    
def log_pos_pdf(sample, l, agg = None):
    log_prob = tfp.distributions.Poisson(rate=l).log_prob(sample)
    if(agg is None):
        return(log_prob)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def log_nb_pdf(sample, r, p, agg = None):
    # log_prob = tfp.distributions.NegativeBinomial(total_count=r, logits=p).log_prob(sample)
    log_prob = tfp.distributions.NegativeBinomial(total_count=r, logits=p).log_prob(sample)
    #log_prob = log_nb(sample, r, p)
    if(agg is None):
        return(log_prob)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def log_zinb_pdf(sample, r, p, d, agg = None):
    log_prob = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=tf.stack([d, 1-d], -1)),
        components=[tfp.distributions.Deterministic(loc = tf.zeros_like(d)), tfp.distributions.NegativeBinomial(total_count = r, logits = p)]).log_prob(sample)

    
    if(agg is None):
        return(log_prob)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function   
def OTDistance(sample, mean, agg = None):
    sample = tf.reshape(sample, [sample.shape[0], -1])
    mean = tf.reshape(mean, [mean.shape[0], -1])
    log_prob = - tf.square(sample - mean)
    if(agg is None):
        return(log_prob)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(log_prob, axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function    
def log_normal_pdf(sample, mean, scale, agg = None):
    log_prob = tfp.distributions.Normal(loc = mean, scale = tf.exp(scale)).log_prob(sample)
    if(agg is None):
        return(log_prob)
    if(not isinstance(agg, (str))):
        return(tf.reduce_mean(log_prob * agg[None, :log_prob.shape[-1]], axis = -1))
    if(agg == 'sum'):
        return(tf.reduce_sum(log_prob, axis = -1))
    return(tf.reduce_mean(log_prob, axis = -1))

@tf.function
def trace_log(Mat):
    return(tf.reduce_mean(tf.math.log(tf.linalg.diag_part(Mat)), axis = -1))

@tf.function
def log_wish_pdf(sample, scale, agg = 'mean'):
    if(agg == 'mean'):
        return(tfp.distributions.WishartTriL(df = sample.shape[-1], scale_tril = scale, input_output_cholesky = True).log_prob(sample)/(sample.shape[-1] ** 2))
    elif(agg == 'mean'):
        return(tfp.distributions.WishartTriL(df = sample.shape[-1], scale_tril = scale, input_output_cholesky = True).log_prob(sample))

# @tf.function
# def Sinkhorn(OT_mat, env_mat, reg = 0.1):
#     M =  tf.reduce_sum(tf.square(OT_mat), axis = 1)[None, :, None] - 2 * tf.matmul(OT_mat, env_mats, transpose_b = True) + tf.reduce_sum(tf.square(env_mats), axis = -1)[:, None, :]
#     M = tf.square(M)
#     M  = tf.transpose(M, [0,2,1])/tf.stop_gradient(tf.reduce_max(M, axis = (1,2))[:, None, None])
#     K = tf.exp(-M/reg)

#     u = np.ones([env_mats.shape[0], env_mats.shape[1]])#/env_mats.shape[1]    

#     for i in range(100):

#         v = (1/env_mats.shape[1])/tf.squeeze(tf.matmul(K, u[:, :, None], transpose_a = True)) #tf.squeeze(tf.matmul(K, u[:, :, None], transpose_a = True)
#         u = (1/env_mats.shape[1])/tf.squeeze(tf.matmul(K, v[:, :, None], transpose_a = False))

#     T = u[:, :, None] * (K * v[:, None, :])

#     return(tf.reduce_sum(T * M, axis = [1,2]))