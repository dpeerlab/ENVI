import tensorflow as tf
import numpy as np
import sklearn.neighbors
import scanpy as sc
import tensorflow_probability as tfp
import time
import sklearn.neural_network
import pandas as pd
import pickle

from scENVI import utils
from scENVI import output_layer


def covet(
    data, k=8, g=64, genes=[], spatial_key="spatial", batch_key=-1, covet_pseudocount=1
):
    """
    Computes COVET matrices for spatial data.
    Adds COVET and COVET_SQRT to data.obsm and COVET_Genes to data.uns

    Args:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        k (int): number of nearest neighbors in niche
        g (int): number of highly variable genes to use for COVET
        genes (list of str): list of genes to use for COVET
            (if empty, uses g highly variable genes)
        batch_key (str): obs key for batch information (default -1, for no batch)
        covet_pseudocount (float): log pseudo-count for COVET computation
            (if 0, use unlogged values)

    Returns:
        updated AnnData object with COVET and COVET_SQRT in obsm and COVET_Genes in uns
    """

    covet, covet_sqrt, _, covet_genes = utils.get_covet(
        data,
        k,
        g,
        genes,
        "OT",
        spatial_key=spatial_key,
        batch_key=batch_key,
        covet_pseudocount=covet_pseudocount,
    )

    data.obsm["COVET"] = covet.astype("float32")
    data.obsm["COVET_SQRT"] = covet_sqrt.astype("float32")
    data.uns["COVET_Genes"] = covet_genes

    return data


class ENVI:

    """
    ENVI integrates spatial and single-cell data

    Parameters:
        spatial_data (anndata): anndata with spatial data, with obsm 'spatial'
            indicating spatial location of spot/segmented cell
        sc_data (anndata): anndata with single cell data
        spatial_key (str): obsm key name with physical location of spots/cells
            (default 'spatial')
        batch_key (str): obs key name of batch/sample of spatial data
            (default 'batch' if exists on .obs, set -1 to ignore)
        num_layers (int): number of layers for decoders and encoders (default 3)
        num_neurons (int): number of neurons in each layer (default 1024)
        latent_dim (int): size of ENVI latent dimension (size 512)
        k_nearest (int): number of nearest neighbors to describe niche (default 8)
        num_covet_genes (int): number of HVGs to compute COVET
            with default (64), if -1 takes all genes
        covet_genes (list of str): manual genes to compute niche with (default [])
        num_HVG (int): number of HVGs to keep for single cell data (default 2048),
            if -1 takes all genes
        sc_genes (list of str): manual genes to keep for single cell data (default [])
        spatial_distribution (str): distribution used to describe spatial data
            (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        sc_distribution (str): distribution used to describe single cell data
            (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        covet_distribution (str): distance metric used for COVET matrices
            (default OT, could be 'OT', 'wish' or 'norm')
        prior_distribution (str): prior distribution for latent (default normal)
        share_dispersion (bool): whether to share dispersion parameters between
            spatial_distribution and sc_distribution (default False)
        const_dispersion (bool): if True, dispersion parameter(s) are only per gene
            rather there per gene per sample (default False)
        spatial_weight (float): coefficient for spatial expression loss in total ELBO
            (default 1.0)
        sc_weight (float): coefficient for single cell expression loss in total ELBO
            (default 1.0)
        covet_weight (float): coefficient for spatial niche loss in total ELBO
            (default 1.0)
        kl_weight (float): coefficient for latent prior loss in total ELBO (default 1.0)
        skip (bool): whether to use skip connections in encoder and decoder networks
            (default True)
        log_input (float): if larger than zero, a log is applied to input with
            pseudocount of log_input (default 0.0)
        covet_pseudocount (float): pseudocount to add to spatial_data when
            computing COVET. Only used when positive. (default 1.0)
        spatial_pseudocount (float): pseudocount to add to spatial_data.
            Only used when positive. (default 1.0)
        sc_pseudocount (float): pseudocount to add to sc_data.
            Only used when positive. (default 1.0)
        library_size (float or Bool) = if true, performs median library size
            if number, normalize library size to it
            if False does nothing (default False)
        z_score (float): if True and spatial/sc_distribution are 'norm' or 'full_norm',
            spatial and single cell data are z-scored (default False)
        agg (str or np.array): aggregation function of loss factors,
            'mean' will average across neurons,
            'sum' will sum across neurons (makes a difference because
            different number of genes for spatial and single cell
            data),
            var will take a per-gene average weighed by elements in anndata.var[var]
        init_scale (float): scale for VarianceScaling normalization of
            initial layer parameters (default 1.0)
        stable (float): pseudocount for rate parameter to stabilize training
            (default 0.0)
    """

    def __init__(
        self,
        spatial_data=None,
        sc_data=None,
        spatial_key="spatial",
        batch_key="batch",
        num_layers=3,
        num_neurons=1024,
        latent_dim=512,
        k_nearest=8,
        num_covet_genes=64,
        covet_genes=[],
        num_HVG=2048,
        sc_genes=[],
        spatial_distribution="pois",
        covet_distribution="OT",
        sc_distribution="nb",
        prior_distribution="norm",
        share_dispersion=False,
        const_dispersion=False,
        spatial_weight=1,
        sc_weight=1,
        covet_weight=1,
        kl_weight=0.3,
        skip=True,
        log_input=0.0,
        covet_pseudocount=1,
        spatial_pseudocount=-1,
        sc_pseudocount=-1,
        z_score=False,
        library_size=False,
        agg="mean",
        init_scale=0.1,
        stable=0.0,
        save_path=None,
        load_path=None,
        **kwargs,
    ):
        super(ENVI, self).__init__()

        if load_path is None:
            self.spatial_data = spatial_data.copy()
            self.sc_data = sc_data.copy()
            self.library_size = library_size

            self.num_layers = num_layers
            self.num_neurons = num_neurons
            self.latent_dim = latent_dim

            self.spatial_distribution = spatial_distribution
            self.covet_distribution = covet_distribution
            self.sc_distribution = sc_distribution
            self.share_dispersion = share_dispersion
            self.const_dispersion = const_dispersion

            self.prior_distribution = prior_distribution

            self.spatial_weight = spatial_weight
            self.sc_weight = sc_weight
            self.covet_weight = covet_weight
            self.kl_weight = kl_weight
            self.skip = skip
            self.agg = agg

            self.num_HVG = num_HVG
            self.sc_genes = sc_genes

            if self.sc_data.raw is None:
                self.sc_data.raw = self.sc_data

            if self.spatial_data.raw is None:
                self.spatial_data.raw = self.spatial_data

            self.overlap_genes = np.asarray(
                np.intersect1d(self.spatial_data.var_names, self.sc_data.var_names)
            )
            self.spatial_data = self.spatial_data[:, list(self.overlap_genes)]

            self.sc_data.layers["log"] = np.log(self.sc_data.X + 1)
            sc.pp.highly_variable_genes(
                self.sc_data,
                n_top_genes=min(self.num_HVG, self.sc_data.shape[-1]),
                layer="log",
            )

            if self.num_HVG == 0:
                self.sc_data.var.highly_variable = False
            if len(self.sc_genes) > 0:
                self.sc_data.var["highly_variable"][
                    np.intersect1d(self.sc_genes, self.sc_data.var_names)
                ] = True

            self.sc_data = self.sc_data[
                :,
                np.union1d(
                    self.sc_data.var_names[self.sc_data.var.highly_variable],
                    self.spatial_data.var_names,
                ),
            ]

            self.non_overlap_genes = np.asarray(
                list(set(self.sc_data.var_names) - set(self.spatial_data.var_names))
            )
            self.sc_data = self.sc_data[
                :, list(self.overlap_genes) + list(self.non_overlap_genes)
            ]

            if self.library_size:
                if isinstance(self.library_size, bool):
                    sc.pp.normalize_total(
                        self.sc_data,
                        target_sum=np.median(self.sc_data.X.sum(axis=1)),
                        inplace=True,
                    )
                else:
                    sc.pp.normalize_total(
                        self.sc_data, target_sum=self.library_size, inplace=True
                    )

            self.k_nearest = k_nearest
            self.spatial_key = spatial_key

            if batch_key in self.spatial_data.obs.columns:
                self.batch_key = batch_key
            else:
                self.batch_key = -1
            print("Computing COVET Matrices")

            self.num_covet_genes = min(num_covet_genes, self.spatial_data.shape[-1])
            self.covet_genes = covet_genes
            self.covet_pseudocount = covet_pseudocount

            (
                self.spatial_data.obsm["COVET"],
                self.spatial_data.obsm["COVET_SQRT"],
                self.spatial_data.obsm["NicheMat"],
                self.covet_genes,
            ) = utils.get_covet(
                self.spatial_data,
                self.k_nearest,
                self.num_covet_genes,
                self.covet_genes,
                self.covet_distribution,
                spatial_key=self.spatial_key,
                batch_key=self.batch_key,
                covet_pseudocount=self.covet_pseudocount,
            )

            self.n_overlap_genes = self.overlap_genes.shape[0]
            self.n_covet_genes = self.spatial_data.obsm["COVET_SQRT"].shape[-1]
            self.n_whole_transcriptome_genes = self.sc_data.shape[-1]

            if (self.agg != "sum") and (self.agg != "mean"):
                self.agg_spatial = self.spatial_data.var[self.agg].astype("float32")
                self.agg_sc = self.sc_data.var[self.agg].astype("float32")
            else:
                self.agg_spatial = self.agg
                self.agg_sc = self.agg

            self.spatial_pseudocount = spatial_pseudocount
            self.sc_pseudocount = sc_pseudocount

            self.log_spatial = False
            self.log_sc = False

            self.z_score = z_score
            self.log_input = (
                log_input
                if log_input > -min(self.sc_data.X.min(), self.sc_data.X.min())
                else 0
            )

            if (
                self.spatial_distribution == "norm"
                or self.spatial_distribution == "full_norm"
                and self.spatial_pseudocount > 0
            ) or (self.spatial_pseudocount > (1 - self.spatial_data.X.min())):
                self.log_spatial = True
                self.spatial_pseudocount = self.spatial_pseudocount
                self.spatial_data.uns["log_pc"] = self.spatial_pseudocount
                self.spatial_data.X = np.log(
                    self.spatial_data.X + self.spatial_data.uns["log_pc"]
                )

            if (
                self.sc_distribution == "norm"
                or self.sc_distribution == "full_norm"
                and self.sc_pseudocount > 0
            ) or (self.sc_pseudocount > (1 - self.sc_data.X.min())):
                self.log_sc = True
                self.sc_pseudocount = self.sc_pseudocount
                self.sc_data.uns["log_pc"] = self.sc_pseudocount
                self.sc_data.X = np.log(self.sc_data.X + self.sc_data.uns["log_pc"])

            self.data_scale = np.abs(self.spatial_data.X).mean()

            if (
                self.z_score
                and (
                    self.sc_distribution == "norm"
                    or self.sc_distribution == "full_norm"
                )
                and (
                    self.spatial_distribution == "norm"
                    or self.spatial_distribution == "full_norm"
                )
            ):
                self.spatial_data.var["mean"] = self.spatial_data.X.mean(axis=0)
                self.spatial_data.var["std"] = self.spatial_data.X.std(axis=0)

                self.spatial_data.X = (
                    self.spatial_data.X - self.spatial_data.var["mean"][None, :]
                ) / self.spatial_data.var["std"][None, :]

                self.sc_data.var["mean"] = self.sc_data.X.mean(axis=0)
                self.sc_data.var["std"] = self.sc_data.X.std(axis=0)

                self.sc_data.X = (
                    self.sc_data.X - self.sc_data.var["mean"][None, :]
                ) / self.sc_data.var["std"][None, :]

                self.data_scale = 1

            self.stable = stable
            self.init_scale = init_scale

            self.encoder_layers = []
            self.decoder_expression_layers = []
            self.decoder_covet_layers = []

            self.initializer_layers = tf.keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=np.sqrt(self.init_scale / self.num_neurons) / self.data_scale,
            )
            self.initializer_encoder = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=np.sqrt(self.init_scale / self.num_neurons)
            )
            self.initializer_output_covet = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=np.sqrt(self.init_scale / self.num_neurons)
            )
            self.initializer_output_expression = tf.keras.initializers.TruncatedNormal(
                mean=0.0, stddev=np.sqrt(self.init_scale / self.overlap_genes.shape[0])
            )

            print("Initializing VAE")

            for i in range(self.num_layers - 1):
                self.encoder_layers.append(
                    tf.keras.layers.Dense(
                        units=self.num_neurons,
                        kernel_initializer=self.initializer_layers,
                        bias_initializer=self.initializer_layers,
                        name="encoder_" + str(i),
                    )
                )

                self.decoder_expression_layers.append(
                    tf.keras.layers.Dense(
                        units=self.num_neurons,
                        kernel_initializer=self.initializer_layers,
                        bias_initializer=self.initializer_layers,
                        name="decoder_expression_" + str(i),
                    )
                )

                self.decoder_covet_layers.append(
                    tf.keras.layers.Dense(
                        units=self.num_neurons,
                        kernel_initializer=self.initializer_layers,
                        bias_initializer=self.initializer_layers,
                        name="decoder_covet_" + str(i),
                    )
                )

            self.encoder_layers.append(
                tf.keras.layers.Dense(
                    units=2 * latent_dim,
                    kernel_initializer=self.initializer_encoder,
                    bias_initializer=self.initializer_encoder,
                    name="encoder_output",
                )
            )

            self.decoder_expression_layers.append(
                output_layer.ENVIOutputLayer(
                    input_dim=self.num_neurons,
                    units=self.n_whole_transcriptome_genes,
                    spatial_distribution=self.spatial_distribution,
                    sc_distribution=self.sc_distribution,
                    share_dispersion=self.share_dispersion,
                    const_dispersion=self.const_dispersion,
                    kernel_init=self.initializer_output_expression,
                    bias_init=self.initializer_output_expression,
                    name="decoder_expression_output",
                )
            )

            self.decoder_covet_layers.append(
                tf.keras.layers.Dense(
                    units=int(self.n_covet_genes * (self.n_covet_genes + 1) / 2),
                    kernel_initializer=self.initializer_output_covet,
                    bias_initializer=self.initializer_output_covet,
                    name="decoder_covet_output",
                )
            )

            if save_path is not None:
                self.save_path = save_path
            else:
                self.save_path = -1
            print("Finished Initializing ENVI")

        else:
            self.load_path = load_path

            with open(self.load_path, "rb") as handle:
                envi_model = pickle.load(handle)

            for key, val in zip(envi_model.keys(), envi_model.values()):
                setattr(self, key, val)

            print("Finished loading ENVI model")

    @tf.function
    def encode_nn(self, input):
        """
        Encoder forward pass

        Args:
            input (array): input to encoder NN
                (size of #genes in spatial data + confounder)
        Returns:
            output (array): NN output
        """

        output = input
        for i in range(self.num_layers - 1):
            output = self.encoder_layers[i](output) + (
                output if (i > 0 and self.skip) else 0
            )
            output = tf.nn.leaky_relu(output)
        return self.encoder_layers[-1](output)

    @tf.function
    def decode_expression_nn(self, input):
        """
        Expression decoder forward pass

        Args:
            input (array): input to expression decoder NN
                (size of latent dimension + confounder)

        Returns:
            output (array): NN output
        """

        output = input
        for i in range(self.num_layers - 1):
            output = self.decoder_expression_layers[i](output) + (
                output if (i > 0 and self.skip) else 0
            )
            output = tf.nn.leaky_relu(output)
        return output

    @tf.function
    def decode_cov_nn(self, input):
        """
        Covariance (niche) decoder forward pass

        Args:
            input (array): input to niche decoder NN
                (size of latent dimension + confounder)

        Returns:
            output (array): NN output
        """
        output = input
        for i in range(self.num_layers - 1):
            output = self.decoder_covet_layers[i](output) + (
                output if (i > 0 and self.skip) else 0
            )
            output = tf.nn.leaky_relu(output)
        return self.decoder_covet_layers[-1](output)

    @tf.function
    def encode(self, x, mode="sc"):
        """
        Appends confounding variable to input and generates an encoding

        Args:
            x (array): input to encoder (size of #genes in spatial data)
            mode (str): 'sc' for single cell, 'spatial' for spatial data

        Return:
            mean (array): mean parameter for latent variable
            log_std (array): log of the standard deviation for latent variable
        """

        confounder = 0 if mode == "spatial" else 1
        if self.log_input > 0:
            x = tf.math.log(x + self.log_input)
        x_confounder = tf.concat(
            [
                x,
                tf.one_hot(
                    confounder * tf.ones(x.shape[0], dtype=tf.uint8),
                    2,
                    dtype=tf.keras.backend.floatx(),
                ),
            ],
            axis=-1,
        )
        return tf.split(self.encode_nn(x_confounder), num_or_size_splits=2, axis=1)

    @tf.function
    def expression_decode(self, x, mode="sc"):
        """
        Appends confounding variable to latent and generates an output distribution

        Args:
            x (array): input to expression decoder (size of latent dimension)
            mode (str): 'sc' for single cell, 'spatial' for spatial data

        Return:
            Output paramterizations for chosen expression distributions
        """
        confounder = 0 if mode == "spatial" else 1
        x_confounder = tf.concat(
            [
                x,
                tf.one_hot(
                    confounder * tf.ones(x.shape[0], dtype=tf.uint8),
                    2,
                    dtype=tf.keras.backend.floatx(),
                ),
            ],
            axis=-1,
        )
        decoder_output = self.decode_expression_nn(x_confounder)

        if getattr(self, mode + "_distribution") == "zinb":
            output_r, output_p, output_d = self.decoder_expression_layers[-1](
                decoder_output, mode
            )

            return (
                tf.nn.softplus(output_r) + self.stable,
                output_p,
                tf.nn.sigmoid(0.01 * output_d - 2),
            )
        if getattr(self, mode + "_distribution") == "nb":
            output_r, output_p = self.decoder_expression_layers[-1](
                decoder_output, mode
            )

            return tf.nn.softplus(output_r) + self.stable, output_p
        if getattr(self, mode + "_distribution") == "pois":
            output_l = self.decoder_expression_layers[-1](decoder_output, mode)
            return tf.nn.softplus(output_l) + self.stable
        if getattr(self, mode + "_distribution") == "full_norm":
            output_mu, output_logstd = self.decoder_expression_layers[-1](
                decoder_output, mode
            )
            return output_mu, output_logstd
        if getattr(self, mode + "_distribution") == "norm":
            output_mu = self.decoder_expression_layers[-1](decoder_output, mode)
            return output_mu

    @tf.function
    def covet_decode(self, x):
        """
        Generates an output distribution for niche data

        Args:
            x (array): input to COVET decoder (size of latent dimension)

        Return:
            Output paramterizations for chosen niche distributions
        """

        DecOut = self.decode_cov_nn(x)
        if self.covet_distribution == "wish":
            TriMat = tfp.math.fill_triangular(DecOut)
            TriMat = tf.linalg.set_diag(
                TriMat, tf.math.softplus(tf.linalg.diag_part(TriMat))
            )
            return TriMat
        elif self.covet_distribution == "norm":
            TriMat = tfp.math.fill_triangular(DecOut)
            return 0.5 * TriMat + 0.5 * tf.tranpose(TriMat, [0, 2, 1])
        elif self.covet_distribution == "OT":
            TriMat = tfp.math.fill_triangular(DecOut)
            return tf.matmul(TriMat, TriMat, transpose_b=True)

    @tf.function
    def encoder_mean(self, mean, logstd):
        """
        Returns posterior mean given latent parametrization, which is not the mean
            variable for a log_normal prior

        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            Posterior mean for latent
        """
        if self.prior_distribution == "norm":
            return mean
        elif self.prior_distribution == "log_norm":
            return tf.exp(mean + tf.square(tf.exp(logstd)) / 2)

    @tf.function
    def reparameterize(self, mean, logstd):
        """
        Samples from latent using the reparameterization trick

        Args:
            mean (array): latent mean parameter
            logstd (array): latent mean parameter
        Return:
            sample from latent
        """
        reparm = (
            tf.random.normal(shape=mean.shape, dtype=tf.keras.backend.floatx())
            * tf.exp(logstd)
            + mean
        )
        if self.prior_distribution == "norm":
            return reparm
        elif self.prior_distribution == "log_norm":
            return tf.exp(reparm)

    @tf.function
    def compute_loss(self, spatial_sample, cov_sample, sc_sample):
        """
        Computes ENVI likelihoods

        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): COVET data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted
                to spatial genes
        Return:
            spatial_likelihood: ENVI likelihood for spatial expression
            covet_likelihood: ENVI likelihood for COVET data
            sc_likelihood: ENVI likelihood for single cell data
            kl: KL divergence between posterior latent and prior
        """
        mean_spatial, logstd_spatial = self.encode(
            spatial_sample[:, : self.n_overlap_genes], mode="spatial"
        )
        mean_sc, logstd_sc = self.encode(
            sc_sample[:, : self.n_overlap_genes], mode="sc"
        )

        z_spatial = self.reparameterize(mean_spatial, logstd_spatial)
        z_sc = self.reparameterize(mean_sc, logstd_sc)

        if self.spatial_distribution == "zinb":
            spatial_r, spatial_p, spatial_d = self.expression_decode(
                z_spatial, mode="spatial"
            )
            spatial_likelihood = tf.reduce_mean(
                utils.log_zinb_pdf(
                    spatial_sample,
                    spatial_r[:, : spatial_sample.shape[-1]],
                    spatial_p[:, : spatial_sample.shape[-1]],
                    spatial_d[:, : spatial_sample.shape[-1]],
                    agg=self.agg_spatial,
                ),
                axis=0,
            )
        if self.spatial_distribution == "nb":
            spatial_r, spatial_p = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = tf.reduce_mean(
                utils.log_nb_pdf(
                    spatial_sample,
                    spatial_r[:, : spatial_sample.shape[-1]],
                    spatial_p[:, : spatial_sample.shape[-1]],
                    agg=self.agg_spatial,
                ),
                axis=0,
            )
        if self.spatial_distribution == "pois":
            spatial_l = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = tf.reduce_mean(
                utils.log_pos_pdf(
                    spatial_sample,
                    spatial_l[:, : spatial_sample.shape[-1]],
                    agg=self.agg_spatial,
                ),
                axis=0,
            )
        if self.spatial_distribution == "full_norm":
            spatial_mu, spatial_logstd = self.expression_decode(
                z_spatial, mode="spatial"
            )
            spatial_likelihood = tf.reduce_mean(
                utils.log_normal_pdf(
                    spatial_sample,
                    spatial_mu[:, : spatial_sample.shape[-1]],
                    spatial_logstd[:, : spatial_sample.shape[-1]],
                    agg=self.agg_spatial,
                ),
                axis=0,
            )
        if self.spatial_distribution == "norm":
            spatial_mu = self.expression_decode(z_spatial, mode="spatial")
            spatial_likelihood = tf.reduce_mean(
                utils.log_normal_pdf(
                    spatial_sample,
                    spatial_mu[:, : spatial_sample.shape[-1]],
                    tf.zeros_like(spatial_sample),
                    agg=self.agg_spatial,
                ),
                axis=0,
            )

        if self.sc_distribution == "zinb":
            sc_r, sc_p, sc_d = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = tf.reduce_mean(
                utils.log_zinb_pdf(sc_sample, sc_r, sc_p, sc_d, agg=self.agg_sc), axis=0
            )
        if self.sc_distribution == "nb":
            sc_r, sc_p = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = tf.reduce_mean(
                utils.log_nb_pdf(sc_sample, sc_r, sc_p, agg=self.agg_sc), axis=0
            )
        if self.sc_distribution == "pois":
            sc_l = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = tf.reduce_mean(
                utils.log_pos_pdf(sc_sample, sc_l, agg=self.agg_sc), axis=0
            )
        if self.sc_distribution == "full_norm":
            sc_mu, sc_std = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = tf.reduce_mean(
                utils.log_normal_pdf(sc_sample, sc_mu, sc_std, agg=self.agg_sc), axis=0
            )
        if self.sc_distribution == "norm":
            sc_mu = self.expression_decode(z_sc, mode="sc")
            sc_likelihood = tf.reduce_mean(
                utils.log_normal_pdf(
                    sc_sample, sc_mu, tf.zeros_like(sc_sample), agg=self.agg_sc
                ),
                axis=0,
            )

        if self.covet_distribution == "wish":
            cov_mu = self.covet_decode(z_spatial)
            covet_likelihood = tf.reduce_mean(
                utils.log_wish_pdf(cov_sample, cov_mu, agg=self.agg), axis=0
            )
        elif self.covet_distribution == "norm":
            cov_mu = tf.reshape(
                self.covet_decode(z_spatial), [spatial_sample.shape[0], -1]
            )
            covet_likelihood = tf.reduce_mean(
                utils.log_normal_pdf(
                    tf.reshape(cov_sample, [cov_sample.shape[0], -1]),
                    cov_mu,
                    tf.zeros_like(cov_mu),
                    agg=self.agg,
                ),
                axis=0,
            )
        elif self.covet_distribution == "OT":
            cov_mu = self.covet_decode(z_spatial)
            covet_likelihood = tf.reduce_mean(
                utils.ot_distance(cov_sample, cov_mu, agg=self.agg), axis=0
            )

        if self.prior_distribution == "norm":
            kl_spatial = tf.reduce_mean(
                utils.normal_kl(mean_spatial, logstd_spatial, agg=self.agg), axis=0
            )
            kl_sc = tf.reduce_mean(
                utils.normal_kl(mean_sc, logstd_sc, agg=self.agg), axis=0
            )
        elif self.prior_distribution == "log_norm":
            kl_spatial = tf.reduce_mean(
                utils.log_normal_kl(mean_spatial, logstd_spatial, agg=self.agg), axis=0
            )
            kl_sc = tf.reduce_mean(
                utils.log_normal_kl(mean_sc, logstd_sc, agg=self.agg), axis=0
            )

        kl = 0.5 * kl_spatial + 0.5 * kl_sc

        return (spatial_likelihood, covet_likelihood, sc_likelihood, kl)

    def get_covet_mean(self, covet):
        """
        Untransforms COVET based on which distribution we are using

        Args:
            covet (array/tensor): transformed COVET matrix to untransform
        Return:
            original COVET matrix
        """
        if self.covet_distribution == "wish":
            return covet * tf.sqrt(covet.shape[-1])
        elif self.covet_distribution == "OT":
            return tf.mamtul(covet, covet)
        else:
            return covet

    def get_mean_sample(self, decode, mode="spatial"):
        """
        Computes mean of expression distribution

        Args:
            decode (list or array/tensor): parameter of distribution
            mode (str): modality of data to compute distribution mean of
                (default 'spatial')
        Return:
            distribution mean from parameterization
        """
        if getattr(self, mode + "_distribution") == "zinb":
            return decode[0] * tf.exp(decode[1]) * (1 - decode[2])
        elif getattr(self, mode + "_distribution") == "nb":
            # return(decode[0])
            return decode[0] * tf.exp(decode[1])
        elif getattr(self, mode + "_distribution") == "pois":
            return decode
        elif getattr(self, mode + "_distribution") == "full_norm":
            return decode[0]
        elif getattr(self, mode + "_distribution") == "norm":
            return decode

    def cluster_rep(self):
        import phenograph

        comm_emb = phenograph.cluster(
            np.concatenate(
                (
                    self.spatial_data.obsm["envi_latent"],
                    self.sc_data.obsm["envi_latent"],
                ),
                axis=0,
            )
        )[0]

        self.spatial_data.obs["latent_cluster"] = comm_emb[: self.spatial_data.shape[0]]
        self.sc_data.obs["latent_cluster"] = comm_emb[self.spatial_data.shape[0] :]

    def latent_rep(self, num_div=16, data=None, mode=None):
        """
        Compute latent embeddings for spatial and single cell data

        Args:
            num_div (int): number of splits for forward pass to allow to fit in gpu
        Return:
            no return, adds 'envi_latent' ENVI.spatial_data.obsm and
            ENVI.spatial_data.obsm
        """

        if data is None:
            self.spatial_data.obsm["envi_latent"] = np.concatenate(
                [
                    self.encode(
                        np.array_split(
                            self.spatial_data.X.astype(tf.keras.backend.floatx()),
                            num_div,
                            axis=0,
                        )[i],
                        mode="spatial",
                    )[0].numpy()
                    for i in range(num_div)
                ],
                axis=0,
            )

            self.sc_data.obsm["envi_latent"] = np.concatenate(
                [
                    self.encode(
                        np.array_split(
                            self.sc_data[:, self.spatial_data.var_names].X.astype(
                                tf.keras.backend.floatx()
                            ),
                            num_div,
                            axis=0,
                        )[i],
                        mode="sc",
                    )[0].numpy()
                    for i in range(num_div)
                ],
                axis=0,
            )

        else:
            data = data.copy()

            if mode == "spatial":
                if not set(self.spatial_data.var_names).issubset(set(data.var_names)):
                    print("(Spatial) Data does not contain trained gene")
                    return -1

                data = data[:, self.spatial_data.var_names]

                if self.log_spatial:
                    data.X = np.log(data.X + self.spatial_data.uns["log_pc"])

                if self.z_score:
                    data.X = (
                        data.X - self.spatial_data.var["mean"]
                    ) / self.spatial_data.var["std"]

            else:
                if not set(self.spatial_data.var_names).issubset(set(data.var_names)):
                    print("(sc) Data does not contain trained gene")
                    return -1

                data = data[:, self.sc_data.var_names]

                if self.log_spatial:
                    data.X = np.log(data.X + self.sc_data.uns["log_pc"])

                if self.z_score:
                    data.X = (data.X - self.sc_data.var["mean"]) / self.sc_data.var[
                        "std"
                    ]

            envi_latent = np.concatenate(
                [
                    self.encode(
                        np.array_split(
                            data[:, self.spatial_data.var_names].X.astype(
                                tf.keras.backend.floatx()
                            ),
                            num_div,
                            axis=0,
                        )[i],
                        mode=mode,
                    )[0].numpy()
                    for i in range(num_div)
                ],
                axis=0,
            )
            return envi_latent

    def pred_type(
        self,
        pred_on="sc",
        key_name="cell_type",
        ClassificationModel=sklearn.neural_network.MLPClassifier(
            alpha=0.01, max_iter=100, verbose=False
        ),
    ):
        """
        Transfer labeling from one modality to the other using latent embeddings

        Args:
            pred_on (str): what modality to predict labeling for
                (default 'sc', i.e. transfer from spatial_data to single cell data)
            key_name (str): obsm key name for labeling (default 'cell_type')
            ClassificationModel (sklearn model): Classification model to
                learn cell labelings (default sklearn.neural_network.MLPClassifier)
        Return:
            no return, adds key_name with cell labelings to ENVI.spatial_data.obsm or
            ENVI.spatial_data.obsm, depending on pred_on
        """

        if pred_on == "sc":
            ClassificationModel.fit(
                self.spatial_data.obsm["envi_latent"], self.spatial_data.obs[key_name]
            )
            self.sc_data.obs[key_name + "_envi"] = ClassificationModel.predict(
                self.sc_data.obsm["envi_latent"]
            )
            self.spatial_data.obs[key_name + "_envi"] = self.spatial_data.obs[key_name]
            print(
                "Finished transferring labels to single cell data! See "
                + key_name
                + "_envi in obsm of ENVI.sc_data"
            )
        else:
            ClassificationModel.fit(
                self.sc_data.obsm["envi_latent"], self.sc_data.obs[key_name]
            )
            self.spatial_data.obs[key_name + "_envi"] = ClassificationModel.predict(
                self.spatial_data.obsm["envi_latent"]
            )
            self.sc_data.obs[key_name + "_envi"] = self.sc_data.obs[key_name]
            print(
                "Finished transferring labels to spatial data! See "
                + key_name
                + "_envi in obsm of ENVI.spatial_data"
            )

    def impute(self, num_div=16, return_raw=True, data=None):
        """
        Input full transcriptome for spatial data

        Args:
            num_div (int): number of splits for forward pass to allow to fit in gpu
            return_raw (bool): if True, un-logs and un-zscores imputation
                if either were chosen
        Return:
            no return, adds 'imputation' to ENVI.spatial_data.obsm
        """

        if data is None:
            decode = np.concatenate(
                [
                    self.get_mean_sample(
                        self.expression_decode(
                            np.array_split(
                                self.spatial_data.obsm["envi_latent"], num_div, axis=0
                            )[i],
                            mode="sc",
                        ),
                        mode="sc",
                    ).numpy()
                    for i in range(num_div)
                ],
                axis=0,
            )

            imputation = pd.DataFrame(
                decode,
                columns=self.sc_data.var_names,
                index=self.spatial_data.obs_names,
            )

            if return_raw:
                if self.z_score:
                    imputation = (
                        imputation * self.sc_data.var["std"] + self.sc_data.var["mean"]
                    )
                if self.log_sc:
                    imputation = np.exp(imputation) - self.sc_data.uns["log_pc"]
                    imputation[imputation < 0] = 0

            self.spatial_data.obsm["imputation"] = imputation
        else:
            latent = self.latent_rep(data=data, mode="spatial")

            decode = np.concatenate(
                [
                    self.get_mean_sample(
                        self.expression_decode(
                            np.array_split(latent, num_div, axis=0)[i], mode="sc"
                        ),
                        mode="sc",
                    ).numpy()
                    for i in range(num_div)
                ],
                axis=0,
            )

            imputation = pd.DataFrame(
                decode,
                columns=self.sc_data.var_names,
                index=self.spatial_data.obs_names,
            )

            if return_raw:
                if self.z_score:
                    imputation = (
                        imputation * self.sc_data.var["std"] + self.sc_data.var["mean"]
                    )
                if self.log_sc:
                    imputation = np.exp(imputation) - self.sc_data.uns["log_pc"]
                    imputation[imputation < 0] = 0

            return imputation

        print(
            "Finished imputing missing gene for spatial data! "
            "See 'imputation' in obsm of ENVI.spatial_data"
        )

    def infer_covet(self, num_div=16, data=None):
        """
        Infer COVET for single cell data

        Args:
            num_div (int): number of splits for forward pass to allow to fit in gpu
            data (anndata): if not None, use this data instead of ENVI.sc_data
        Return:
            no return, adds 'COVET_SQRT' or 'COVET' to ENVI.sc_data.obsm
        """

        if data is None:
            self.sc_data.obsm["COVET_SQRT"] = np.concatenate(
                [
                    self.covet_decode(
                        np.array_split(
                            self.sc_data.obsm["envi_latent"], num_div, axis=0
                        )[i]
                    )
                    for i in range(num_div)
                ],
                axis=0,
            )
            self.sc_data.obsm["COVET"] = np.concatenate(
                [
                    np.linalg.matrix_power(
                        np.array_split(
                            self.sc_data.obsm["COVET_SQRT"], num_div, axis=0
                        )[i],
                        2,
                    )
                    for i in range(num_div)
                ],
                axis=0,
            )
        else:
            latent = self.latent_rep(data=data, mode="sc")
            covet_sqrt = np.concatenate(
                [
                    self.covet_decode(np.array_split(latent, num_div, axis=0)[i])
                    for i in range(num_div)
                ],
                axis=0,
            )
            covet = np.concatenate(
                [
                    np.linalg.matrix_power(
                        np.array_split(covet_sqrt, num_div, axis=0)[i], 2
                    )
                    for i in range(num_div)
                ],
                axis=0,
            )
            return (covet_sqrt, covet)

    def infer_covet_spatial(self, num_div=16):
        """
        Reconstruct COVET for spatial data

        Args:
            num_div (int): number of splits for forward pass to allow to fit in gpu

        Return:
            no return, adds 'COVET_SQRT' or 'COVET' to ENVI.sc_data.obsm
        """

        self.spatial_data.obsm["COVET_SQRT_envi"] = np.concatenate(
            [
                self.covet_decode(
                    np.array_split(
                        self.spatial_data.obsm["envi_latent"], num_div, axis=0
                    )[i]
                )
                for i in range(num_div)
            ],
            axis=0,
        )
        self.spatial_data.obsm["COVET_envi"] = np.concatenate(
            [
                np.linalg.matrix_power(
                    np.array_split(
                        self.spatial_data.obsm["COVET_SQRT_envi"], num_div, axis=0
                    )[i],
                    2,
                )
                for i in range(num_div)
            ],
            axis=0,
        )

    def reconstruct_niche(
        self,
        k=8,
        niche_key="cell_type",
        pred_key=None,
        norm_reg=False,
        cluster=False,
        res=0.5,
        data=None,
    ):
        """
        Infer niche composition for single cell data
        and add 'niche_by_type' to ENVI.sc_data.obsm

        Args:
            k (float): k for kNN regression on COVET matrices (default 32)
            niche_key (str): spatial obsm key to reconstruct niche from
                (default 'cell_type')
            pred_key (str): spatial & single cell obsm key to split up kNN regression by
                (default None)
            gpu (bool): if True, uses gpu for kNN regression (default False)
            norm_reg (bool): if True, cell type enrichment in normalized by the number
                of cells per type (default False)
            cluster (bool): if True, clusters COVET and produces niche based
                on average across cluster, k is parameter for phenograph (default False)
            res (float): resolution parameter for leiden clustering in phenograph
                (default 0.5)
        """

        #        print(cluster)

        import sklearn.preprocessing

        label_encoding = sklearn.preprocessing.LabelBinarizer().fit(
            self.spatial_data.obs[niche_key]
        )
        spatial_cell_type_encoding = label_encoding.transform(
            self.spatial_data.obs[niche_key]
        )
        self.spatial_data.obsm[niche_key + "_encoding"] = spatial_cell_type_encoding
        cell_type_name = label_encoding.classes_

        neighbor_cell_type = utils.get_niche_expression(
            self.spatial_data,
            self.k_nearest,
            data_key=(niche_key + "_encoding"),
            spatial_key=self.spatial_key,
            batch_key=self.batch_key,
        )

        neighbor_cell_type = neighbor_cell_type.sum(axis=1).astype("float32")

        self.spatial_data.obsm["niche_by_type"] = pd.DataFrame(
            neighbor_cell_type,
            columns=cell_type_name,
            index=self.spatial_data.obs_names,
        )

        if cluster:
            import phenograph

        if data is None:
            self.infer_covet(16, False)

            if pred_key is None:
                covet_fit = self.spatial_data.obsm["COVET_SQRT"]
                covet_pred = self.sc_data.obsm["COVET_SQRT"]

                neighbor_fit = self.spatial_data.obsm["niche_by_type"]

                if norm_reg:
                    neighbor_fit = neighbor_fit / self.spatial_data.obsm[
                        "cell_type_encoding"
                    ].sum(axis=0, keepdims=True)
                    neighbor_fit = (
                        neighbor_fit
                        / neighbor_fit.sum(axis=1, keepdims=True)
                        * self.k_nearest
                    )

                if cluster:
                    with utils.HiddenPrints():
                        phenoclusters = phenograph.cluster(
                            np.concatenate(
                                (
                                    covet_fit.reshape([covet_fit.shape[0], -1]),
                                    covet_pred.reshape([covet_pred.shape[0], -1]),
                                ),
                                axis=0,
                            ),
                            clustering_algo="leiden",
                            k=k,
                            resolution_parameter=res,
                        )

                    phenoclusters_fit = phenoclusters[: covet_fit.shape[0]]
                    phenoclusters_pred = phenoclusters[covet_fit.shape[0] :]

                    avg_niche = np.asarray(
                        [
                            neighbor_fit[phenoclusters_fit == clust].mean(axis=0)
                            for clust in np.arange(phenoclusters.max() + 1)
                        ]
                    )
                    pred_niche = avg_niche[phenoclusters_pred]

                    self.sc_data.obsm["niche_by_type"] = pd.DataFrame(
                        pred_niche, columns=cell_type_name, index=self.sc_data.obs_names
                    )
                else:
                    import sklearn.neighbors

                    regressor = sklearn.neighbors.KNeighborsRegressor(
                        n_neighbors=min(k, covet_fit.shape[0])
                    )

                    regressor.fit(
                        covet_fit.reshape([covet_fit.shape[0], -1]), neighbor_fit
                    )
                    predicted_niche = regressor.predict(
                        covet_pred.reshape([covet_pred.shape[0], -1])
                    )
                    self.sc_data.obsm["niche_by_type"] = pd.DataFrame(
                        predicted_niche,
                        columns=cell_type_name,
                        index=self.sc_data.obs_names,
                    )
            else:
                predicted_niche = np.zeros(
                    shape=(self.sc_data.shape[0], neighbor_cell_type.shape[-1])
                )
                for val in np.unique(self.sc_data.obs[pred_key]):
                    covet_fit = self.spatial_data.obsm["COVET_SQRT"][
                        self.spatial_data.obs[pred_key] == val
                    ]
                    covet_pred = self.sc_data.obsm["COVET_SQRT"][
                        self.sc_data.obs[pred_key] == val
                    ]

                    neighbor_fit = np.asarray(
                        self.spatial_data.obsm["niche_by_type"][
                            self.spatial_data.obs[pred_key] == val
                        ]
                    )

                    if norm_reg:
                        neighbor_fit = neighbor_fit / self.spatial_data.obsm[
                            "cell_type_encoding"
                        ].sum(axis=0, keepdims=True)
                        neighbor_fit = (
                            neighbor_fit
                            / neighbor_fit.sum(axis=1, keepdims=True)
                            * self.k_nearest
                        )

                    if cluster:
                        with utils.HiddenPrints():
                            phenoclusters = phenograph.cluster(
                                np.concatenate(
                                    (
                                        covet_fit.reshape([covet_fit.shape[0], -1]),
                                        covet_pred.reshape([covet_pred.shape[0], -1]),
                                    ),
                                    axis=0,
                                ),
                                clustering_algo="leiden",
                                k=k,
                                resolution_parameter=res,
                            )[0]

                        phenoclusters_fit = phenoclusters[: covet_fit.shape[0]]
                        phenoclusters_pred = phenoclusters[covet_fit.shape[0] :]

                        avg_niche = np.asarray(
                            [
                                neighbor_fit[phenoclusters_fit == clust].mean(axis=0)
                                for clust in np.arange(phenoclusters.max() + 1)
                            ]
                        )
                        predicted_niche[self.sc_data.obs[pred_key] == val] = avg_niche[
                            phenoclusters_pred
                        ]

                    else:
                        import sklearn.neighbors

                        regressor = sklearn.neighbors.KNeighborsRegressor(
                            n_neighbors=min(k, covet_fit.shape[0])
                        )

                        regressor.fit(
                            covet_fit.reshape([covet_fit.shape[0], -1]), neighbor_fit
                        )
                        predicted_niche[
                            self.sc_data.obs[pred_key] == val
                        ] = regressor.predict(
                            covet_pred.reshape([covet_pred.shape[0], -1])
                        )

                self.sc_data.obsm["niche_by_type"] = pd.DataFrame(
                    predicted_niche,
                    columns=cell_type_name,
                    index=self.sc_data.obs_names,
                )
        else:
            sc_covet = self.infer_covet(16, False, data)

            if pred_key is None:
                covet_fit = self.spatial_data.obsm["COVET_SQRT"]
                covet_pred = sc_covet

                neighbor_fit = self.spatial_data.obsm["niche_by_type"]

                if norm_reg:
                    neighbor_fit = neighbor_fit / self.spatial_data.obsm[
                        "cell_type_encoding"
                    ].sum(axis=0, keepdims=True)
                    neighbor_fit = (
                        neighbor_fit
                        / neighbor_fit.sum(axis=1, keepdims=True)
                        * self.k_nearest
                    )

                if cluster:
                    with utils.HiddenPrints():
                        phenoclusters = phenograph.cluster(
                            np.concatenate(
                                (
                                    covet_fit.reshape([covet_fit.shape[0], -1]),
                                    covet_pred.reshape([covet_pred.shape[0], -1]),
                                ),
                                axis=0,
                            ),
                            clustering_algo="leiden",
                            k=k,
                            resolution_parameter=res,
                        )

                    phenoclusters_fit = phenoclusters[: covet_fit.shape[0]]
                    phenoclusters_pred = phenoclusters[covet_fit.shape[0] :]

                    avg_niche = np.asarray(
                        [
                            neighbor_fit[phenoclusters_fit == clust].mean(axis=0)
                            for clust in np.arange(phenoclusters.max() + 1)
                        ]
                    )
                    pred_niche = avg_niche[phenoclusters_pred]

                    niche_by_type = pd.DataFrame(
                        pred_niche, columns=cell_type_name, index=self.sc_data.obs_names
                    )
                else:
                    import sklearn.neighbors

                    regressor = sklearn.neighbors.KNeighborsRegressor(
                        n_neighbors=min(k, covet_fit.shape[0])
                    )

                    regressor.fit(
                        covet_fit.reshape([covet_fit.shape[0], -1]), neighbor_fit
                    )
                    predicted_niche = regressor.predict(
                        covet_pred.reshape([covet_pred.shape[0], -1])
                    )
                    niche_by_type = pd.DataFrame(
                        predicted_niche, columns=cell_type_name, index=data.obs_names
                    )
            else:
                predicted_niche = np.zeros(
                    shape=(data.shape[0], neighbor_cell_type.shape[-1])
                )
                for val in np.unique(self.sc_data.obs[pred_key]):
                    covet_fit = self.spatial_data.obsm["COVET_SQRT"][
                        self.spatial_data.obs[pred_key] == val
                    ]
                    covet_pred = sc_covet[data.obs[pred_key] == val]

                    neighbor_fit = np.asarray(
                        self.spatial_data.obsm["niche_by_type"][
                            self.spatial_data.obs[pred_key] == val
                        ]
                    )

                    if norm_reg:
                        neighbor_fit = neighbor_fit / self.spatial_data.obsm[
                            "cell_type_encoding"
                        ].sum(axis=0, keepdims=True)
                        neighbor_fit = (
                            neighbor_fit
                            / neighbor_fit.sum(axis=1, keepdims=True)
                            * self.k_nearest
                        )

                    if cluster:
                        with utils.HiddenPrints():
                            phenoclusters = phenograph.cluster(
                                np.concatenate(
                                    (
                                        covet_fit.reshape([covet_fit.shape[0], -1]),
                                        covet_pred.reshape([covet_pred.shape[0], -1]),
                                    ),
                                    axis=0,
                                ),
                                clustering_algo="leiden",
                                k=k,
                                resolution_parameter=res,
                            )[0]

                        phenoclusters_fit = phenoclusters[: covet_fit.shape[0]]
                        phenoclusters_pred = phenoclusters[covet_fit.shape[0] :]

                        avg_niche = np.asarray(
                            [
                                neighbor_fit[phenoclusters_fit == clust].mean(axis=0)
                                for clust in np.arange(phenoclusters.max() + 1)
                            ]
                        )
                        predicted_niche[data.obs[pred_key] == val] = avg_niche[
                            phenoclusters_pred
                        ]

                    else:
                        import sklearn.neighbors

                        regressor = sklearn.neighbors.KNeighborsRegressor(
                            n_neighbors=min(k, covet_fit.shape[0])
                        )

                        regressor.fit(
                            covet_fit.reshape([covet_fit.shape[0], -1]), neighbor_fit
                        )
                        predicted_niche[data.obs[pred_key] == val] = regressor.predict(
                            covet_pred.reshape([covet_pred.shape[0], -1])
                        )

                niche_by_type = pd.DataFrame(
                    predicted_niche, columns=cell_type_name, index=data.obs_names
                )

            return niche_by_type

    #         print("Finished Niche Reconstruction! "
    #               "See 'niche_by_type' in obsm of ENVI.sc_data")

    @tf.function
    def compute_apply_gradients(self, spatial_sample, cov_sample, sc_sample):
        """
        Applies gradient descent step given training batch

        Args:
            spatial_sample (np.array or tf.tensor): spatial expression data sample/batch
            cov_sample (np.array or tf.tensor): COVET data sample/batch
            sc_sample (np.array or tf.tensor): single cell data sample/batch subsetted
                to spatial genes
        Return:
            spatial_likelihood: ENVI likelihood for spatial expression
            covet_likelihood: ENVI likelihood for COVET data
            sc_likelihood: ENVI likelihood for single cell data
            kl: KL divergence between posterior latent and prior
            nan: True if any factor in loss was nan and doesn't apply gradients
        """

        with tf.GradientTape() as tape:
            spatial_likelihood, covet_likelihood, sc_likelihood, kl = self.compute_loss(
                spatial_sample, cov_sample, sc_sample
            )
            loss = (
                -self.spatial_weight * spatial_likelihood
                - self.sc_weight * sc_likelihood
                - self.covet_weight * covet_likelihood
                + 2 * self.kl_weight * kl
            )

        if not hasattr(ENVI, "trainable_variables"):
            self.trainable_variables = []
            for _, var in enumerate(
                self.encoder_layers
                + self.decoder_expression_layers
                + self.decoder_covet_layers
            ):
                self.trainable_variables = self.trainable_variables + var.weights

        gradients = tape.gradient(loss, self.trainable_variables)
        nan = False

        #         for grad in gradients:
        #             if(tf.reduce_sum(tf.cast(tf.math.is_nan(grad), tf.int8)) > 0):
        #                 nan = True
        # if(tf.math.is_nan(loss)):
        #     nan = True

        if nan:
            return (spatial_likelihood, covet_likelihood, sc_likelihood, kl, True)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return (spatial_likelihood, covet_likelihood, sc_likelihood, kl, False)

    def train(
        self,
        lr=0.0001,
        batch_size=512,
        epochs=np.power(2, 14),
        verbose=64,
        lr_schedule=True,
    ):
        """
        ENVI training loop and computing of latent representation at end

        Args:
            lr (float): learning rate for training (default 0.0001)
            batch_size (int): total batch size for traning, to sample from single cell
                and spatial data (default 512)
            epochs (int): number of training steps (default 16384)
            split (float): train/test split of data (default 1.0)
            verbose (int): how many training step between each print statement,
                if -1 than no printing (default 128)
            lr_schedule (bool): if True, decreases learning rate by factor of 10 after
                0.75 * epochs steps (default True)
        Return:
            no return, trains ENVI and adds 'envi_latent' to
            obsm of ENVI.sc_data and ENVI.spatial_data
        """

        print("Training ENVI for {} steps".format(epochs))

        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        spatial_data_train = self.spatial_data.X.astype(tf.keras.backend.floatx())
        cov_data_train = self.spatial_data.obsm["COVET_SQRT"].astype(
            tf.keras.backend.floatx()
        )
        sc_data_train = self.sc_data.X.astype(tf.keras.backend.floatx())

        ## Run Dummy:
        utils.log_pos_pdf(tf.ones([5], tf.float32), tf.ones([5], tf.float32))

        time.time()

        from tqdm import trange

        tq = trange(epochs, leave=True, desc="")
        for epoch in tq:  # range(0, epochs + 1):
            if (epoch == int(epochs * 0.75)) and lr_schedule:
                self.lr = lr * 0.1
                self.optimizer.lr.assign(self.lr)

            self.batch_spatial = np.random.choice(
                np.arange(spatial_data_train.shape[0]),
                min(batch_size, spatial_data_train.shape[0]),
                replace=False,
            )

            self.batch_sc = np.random.choice(
                np.arange(sc_data_train.shape[0]),
                min(batch_size, sc_data_train.shape[0]),
                replace=False,
            )

            if (epoch % verbose == 0) and (verbose > 0):
                time.time()

                loss_spatial, loss_cov, loss_sc, loss_kl = self.compute_loss(
                    spatial_data_train[self.batch_spatial],
                    cov_data_train[self.batch_spatial],
                    sc_data_train[self.batch_sc],
                )

                print_statement = (
                    "Trn: spatial Loss: {:.5f}, SC Loss: {:.5f}, "
                    "Cov Loss: {:.5f}, KL Loss: {:.5f}"
                ).format(
                    loss_spatial.numpy(),
                    loss_sc.numpy(),
                    loss_cov.numpy(),
                    loss_kl.numpy(),
                )
                tq.set_description(print_statement, refresh=True)

                time.time()

            (
                loss_spatial,
                loss_cov,
                loss_sc,
                loss_kl,
                nan,
            ) = self.compute_apply_gradients(
                spatial_data_train[self.batch_spatial],
                cov_data_train[self.batch_spatial],
                sc_data_train[self.batch_sc],
            )

        print(
            "Finished Training ENVI! - calculating latent embedding, see 'envi_latent' "
            "obsm of ENVI.sc_data and ENVI.spatial_data"
        )

        self.latent_rep()
        self.save_model()

    def save_model(self):
        if self.save_path != -1:
            attribute_name_list = [
                "spatial_data",
                "sc_data",
                "spatial_key",
                "batch_key",
                "num_layers",
                "num_neurons",
                "latent_dim",
                "k_nearest",
                "num_covet_genes",
                "n_overlap_genes",
                "n_covet_genes",
                "n_whole_transcriptome_genes",
                "log_spatial",
                "log_sc",
                "covet_genes",
                "num_HVG",
                "sc_genes",
                "spatial_distribution",
                "covet_distribution",
                "sc_distribution",
                "prior_distribution",
                "share_dispersion",
                "const_dispersion",
                "spatial_weight",
                "sc_weight",
                "kl_weight",
                "skip",
                "log_input",
                "covet_pseudocount",
                "spatial_pseudocount",
                "sc_pseudocount",
                "z_score",
                "library_size",
                "agg",
                "init_scale",
                "stable",
                "encoder_layers",
                "decoder_expression_layers",
                "decoder_covet_layers",
            ]

            attribute_list = {attr: getattr(self, attr) for attr in attribute_name_list}

            directory = self.save_path
            directory = "/".join(directory.split("/")[:-1])

            import os

            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(self.save_path, "wb") as handle:
                pickle.dump(attribute_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
