from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scanpy as sc
import sklearn.neighbors
import tensorflow_probability.substrates.jax as jax_prob # type: ignore
from flax import linen as nn
from jax import jit, random
from tqdm import trange, tqdm

from scenvi._dists import (
    KL,
    AOT_Distance,
    log_nb_pdf,
    log_normal_pdf,
    log_pos_pdf,
    log_zinb_pdf,
)

from scenvi.utils import CVAE, Metrics, TrainState, compute_covet, niche_cell_type


class ENVI:
    """
    Initializes the ENVI model & computes COVET for spatial data

    :param spatial_data: (anndata) spatial transcriptomics data, with an obsm indicating spatial location of spot/segmented cell
    :param sc_data: (anndata) complementary sinlge cell data
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)
    :param num_layers: (int) number of neural network for decoders and encoders (default 3)
    :param num_neurons: (int) number of neurons in each layer (default 1024)
    :param latent_dim: (int) size of ENVI latent dimention (size 512)
    :param k_nearest: (int) number of physical neighbours to describe niche (default 8)
    :param covet_batch_size: (int) batch size for COVET computation (default 256)
    :param num_cov_genes: (int) number of HVGs to compute niche covariance with (default ֿ64), if -1 uses all genes
    :param cov_genes: (list of str) manual genes to compute niche with (default None)
    :param num_HVG: (int) number of HVGs to keep for single cell data (default 2048)
    :param sc_genes: (list of str) manual genes to keep for sinlge cell data (default None)
    :param spatial_dist: (str) distribution used to describe spatial data (default pois, could be 'pois', 'nb', 'zinb' or 'norm')
    :param sc_dist: (str) distribution used to describe sinlge cell data (default nb, could be 'pois', 'nb', 'zinb' or 'norm')
    :param spatial_coeff: (float) coefficient for spatial expression loss in total ELBO (default 1.0)
    :param sc_coeff: (float) coefficient for sinlge cell expression loss in total ELBO (default 1.0)
    :param cov_coeff: (float) coefficient for spatial niche loss in total ELBO (default 1.0)
    :param kl_coeff: (float) coefficient for latent prior loss in total ELBO (default 1.0)
    :param log_input: (float) if larger than zero, a log is applied to ENVI input with pseudocount of log_input (default 0.1)
    :param stable_eps: (float) added value to log probabilty calculations to avoid NaNs during training (default 1e-6)
    :param covet_use_obsm: (str) obsm key to use for COVET calculation instead of gene expression (default None)
    :param covet_use_layer: (str) layer to use for COVET calculation instead of log-transformed X (default None)

    :return: initialized ENVI model
    """

    def __init__(
        self,
        spatial_data,
        sc_data,
        spatial_key="spatial",
        batch_key="batch",
        num_layers=3,
        num_neurons=1024,
        latent_dim=512,
        k_nearest=8,
        covet_batch_size=256,
        num_cov_genes=64,
        cov_genes=None,
        num_HVG=2048,
        sc_genes=None,
        spatial_dist="pois",
        sc_dist="nb",
        spatial_coeff=1,
        sc_coeff=1,
        cov_coeff=1,
        kl_coeff=0.3,
        log_input=0.1,
        stable_eps=1e-6,
        covet_use_obsm=None,
        covet_use_layer=None,
    ):
        
        self.spatial_data, self.sc_data, self.overlap_genes, self.non_overlap_genes = self._prepare_gene_sets(
            spatial_data, sc_data, num_HVG, sc_genes
        )

        if batch_key not in spatial_data.obs.columns:
            batch_key = -1

        self.k_nearest = k_nearest
        self.spatial_key = spatial_key
        self.batch_key = batch_key
        self.cov_genes = cov_genes
        self.num_cov_genes = min(num_cov_genes, self.spatial_data.shape[1])
        self.covet_use_obsm = covet_use_obsm
        self.covet_use_layer = covet_use_layer

        print("Computing Niche Covariance Matrices")

        # Add information about which data source is being used for COVET
        if self.covet_use_obsm is not None:
            print(f"Using obsm '{self.covet_use_obsm}' for COVET calculation")
        elif self.covet_use_layer is not None:
            print(f"Using layer '{self.covet_use_layer}' for COVET calculation")
        else:
            print("Using log-transformed gene expression for COVET calculation")

        (
            self.spatial_data.obsm["COVET"],
            self.spatial_data.obsm["COVET_SQRT"],
            self.CovGenes,
        ) = compute_covet(
            spatial_data=self.spatial_data,
            k=self.k_nearest,
            g=self.num_cov_genes,
            genes=self.cov_genes,
            spatial_key=self.spatial_key,
            batch_key=self.batch_key,
            batch_size=covet_batch_size,
            use_obsm=self.covet_use_obsm,
            use_layer=self.covet_use_layer
        )

        self.overlap_num = self.overlap_genes.shape[0]
        self.cov_gene_num = self.spatial_data.obsm["COVET_SQRT"].shape[-1]
        self.full_trans_gene_num = self.sc_data.shape[-1]

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.latent_dim = latent_dim

        self.spatial_dist = spatial_dist
        self.sc_dist = sc_dist

        self.dist_size_dict = {"pois": 1, "nb": 2, "zinb": 3, "norm": 1}

        self.exp_dec_size = (
            self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]
            + (self.dist_size_dict[self.spatial_dist] - 1) * self.spatial_data.shape[-1]
        )

        self.spatial_coeff = spatial_coeff
        self.sc_coeff = sc_coeff
        self.cov_coeff = cov_coeff
        self.kl_coeff = kl_coeff

        if self.sc_dist == "norm" or self.spatial_dist == "norm" or self.spatial_data.X.min()<0 or self.sc_data.X.min()<0:
            self.log_input = -1
        else:
            self.log_input = log_input

        self.eps = stable_eps

        self.model = CVAE(
            n_layers=self.num_layers,
            n_neurons=self.num_neurons,
            n_latent=self.latent_dim,
            n_output_exp=self.exp_dec_size,
            n_output_cov=int(self.cov_gene_num * (self.cov_gene_num + 1) / 2),
        )

        print("Finished Initializing ENVI")

    def inp_log_fn(self, x):
        """
        :meta private:
        """

        if self.log_input > 0:
            return jnp.log(x + self.log_input)
        return x

    def _prepare_gene_sets(self, spatial_data, sc_data, num_HVG, user_provided_genes=None):
        """
        :meta private:
        """
        print("Preparing gene sets for ENVI analysis...")
        
        
        # Make copies to avoid modifying inputs
        spatial_obj = spatial_data.copy()
        sc_obj = sc_data.copy()
        
        # First restrict spatial data to genes also in single-cell data
        common_genes = np.intersect1d(spatial_obj.var_names, sc_obj.var_names)
        spatial_obj = spatial_obj[:, common_genes]
        
        # Identify highly variable genes if not already done
        if "highly_variable" not in sc_obj.var.columns:
            log_layer = None
            # Find or create appropriate log-transformed layer
            if 'log' in sc_obj.layers.keys():
                log_layer = "log"
            elif 'log1p' in sc_obj.layers.keys():
                log_layer = "log1p"
            elif sc_obj.X.min() < 0:
                # Data already log-transformed
                pass
            else:
                # Create log-transformed layer
                sc_obj.layers["log"] = np.log(sc_obj.X + 1)
                log_layer = "log"
                    
            # Identify highly variable genes
            if log_layer:
                sc.pp.highly_variable_genes(sc_obj, n_top_genes=num_HVG, layer=log_layer)
            else:
                sc.pp.highly_variable_genes(sc_obj, n_top_genes=num_HVG)
            
            print(f"Identified {num_HVG} highly variable genes from single-cell data")
        else:
            print("Using pre-computed highly variable genes from single-cell data")
        
        # Store raw data if not already s tored
        if sc_obj.raw is None:
            sc_obj.raw = sc_obj
                
        # Determine genes to keep in single-cell data
        hvg_genes = sc_obj.var_names[sc_obj.var.highly_variable]
        genes_to_keep = np.union1d(hvg_genes, spatial_obj.var_names)
        
        # Add user-provided genes if any
        if user_provided_genes and len(user_provided_genes) > 0:
            genes_to_keep = np.union1d(genes_to_keep, user_provided_genes)
                
        # Subset single-cell data to genes of interest
        sc_obj = sc_obj[:, genes_to_keep]
        
        # Identify overlap and non-overlap genes
        overlap_genes = np.intersect1d(spatial_obj.var_names, sc_obj.var_names)
        non_overlap_genes = np.setdiff1d(sc_obj.var_names, spatial_obj.var_names)
        
        print(f"Gene selection: {len(overlap_genes)} shared genes, {len(non_overlap_genes)} unique to single-cell")
        
        # Final filtering of both datasets
        spatial_obj = spatial_obj[:, overlap_genes]
        sc_obj = sc_obj[:, np.concatenate([overlap_genes, non_overlap_genes])]
        
        return spatial_obj, sc_obj, overlap_genes, non_overlap_genes
    
    def mean_sc(self, sc_inp):
        """
        :meta private:
        """

        sc_inp = sc_inp[:, : self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]]
        if self.sc_dist == "zinb":
            sc_r, sc_p, sc_d = jnp.split(sc_inp, 3, axis=-1)
            return nn.softplus(sc_r) * jnp.exp(sc_p) * (1 - nn.sigmoid(sc_d))
        if self.sc_dist == "nb":
            sc_r, sc_p = jnp.split(sc_inp, 2, axis=-1)
            return nn.softplus(sc_r) * jnp.exp(sc_p)
        if self.sc_dist == "pois":
            sc_l = sc_inp
            return sc_l
        if self.sc_dist == "norm":
            sc_l = sc_inp
            return sc_l

    def mean_spatial(self, spatial_inp):
        """
        :meta private:
        """

        if self.spatial_dist == "zinb" or self.spatial_dist == "nb":
            spatial_inp = jnp.concatenate(
                [
                    spatial_inp[:, : self.spatial_data.shape[-1]],
                    spatial_inp[
                        :,
                        -(self.dist_size_dict[self.spatial_dist] - 1)
                        * self.spatial_data.shape[-1] :,
                    ],
                ],
                axis=-1,
            )
        else:
            spatial_inp = spatial_inp[:, : self.spatial_data.shape[-1]]

        if self.spatial_dist == "zinb":
            spatial_r, spatial_p, spatial_d = jnp.split(spatial_inp, 3, axis=-1)
            return (
                nn.softplus(spatial_r)
                * jnp.exp(spatial_p)
                * (1 - nn.sigmoid(spatial_d))
            )
        if self.spatial_dist == "nb":
            spatial_r, spatial_p = jnp.split(spatial_inp, 2, axis=-1)
            return nn.softplus(spatial_r) * jnp.exp(spatial_p)
        if self.spatial_dist == "pois":
            spatial_l = spatial_inp
            return spatial_l
        if self.spatial_dist == "norm":
            spatial_l = spatial_inp
            return spatial_l

    def factor_sc(self, sc_inp, dec_exp):
        """
        :meta private:
        """

        sc_neurons = dec_exp[
            :, : self.dist_size_dict[self.sc_dist] * self.sc_data.shape[-1]
        ]

        if self.sc_dist == "zinb":
            sc_r, sc_p, sc_d = jnp.split(sc_neurons, 3, axis=-1)
            sc_like = jnp.mean(
                log_zinb_pdf(sc_inp, nn.softplus(sc_r) + self.eps, sc_p, sc_d)
            )
        if self.sc_dist == "nb":
            sc_r, sc_p = jnp.split(sc_neurons, 2, axis=-1)
            sc_like = jnp.mean(log_nb_pdf(sc_inp, nn.softplus(sc_r) + self.eps, sc_p))
        if self.sc_dist == "pois":
            sc_l = sc_neurons
            sc_like = jnp.mean(log_pos_pdf(sc_inp, nn.softplus(sc_l) + self.eps))
        if self.sc_dist == "norm":
            sc_l = sc_neurons
            sc_like = jnp.mean(log_normal_pdf(sc_inp, sc_l))
        return sc_like

    def factor_spatial(self, spatial_inp, dec_exp):
        """
        :meta private:
        """

        if self.spatial_dist == "zinb" or self.spatial_dist == "nb":
            spatial_neurons = jnp.concatenate(
                [
                    dec_exp[:, : self.spatial_data.shape[-1]],
                    dec_exp[
                        :,
                        -(self.dist_size_dict[self.spatial_dist] - 1)
                        * self.spatial_data.shape[-1] :,
                    ],
                ],
                axis=-1,
            )
        else:
            spatial_neurons = dec_exp[:, : self.spatial_data.shape[-1]]

        if self.spatial_dist == "zinb":
            spatial_r, spatial_p, spatial_d = jnp.split(spatial_neurons, 3, axis=-1)
            spatial_like = jnp.mean(
                log_zinb_pdf(
                    spatial_inp, nn.softplus(spatial_r) + self.eps, spatial_p, spatial_d
                )
            )
        if self.spatial_dist == "nb":
            spatial_r, spatial_p = jnp.split(spatial_neurons, 2, axis=-1)
            spatial_like = jnp.mean(
                log_nb_pdf(spatial_inp, nn.softplus(spatial_r) + self.eps, spatial_p)
            )
        if self.spatial_dist == "pois":
            spatial_l = spatial_neurons
            spatial_like = jnp.mean(
                log_pos_pdf(spatial_inp, nn.softplus(spatial_l) + self.eps)
            )
        if self.spatial_dist == "norm":
            spatial_l = spatial_neurons
            spatial_like = jnp.mean(log_normal_pdf(spatial_inp, spatial_l))
        return spatial_like

    def grammian_cov(self, dec_cov):
        """
        :meta private:
        """

        dec_cov = jax_prob.math.fill_triangular(dec_cov)
        return jnp.matmul(dec_cov, dec_cov.transpose([0, 2, 1]))

    def create_train_state(self, key=random.key(0), init_lr=0.0001, decay_steps=4000):
        """
        :meta private:
        """

        key, subkey1, subkey2 = random.split(key, num=3)
        params = self.model.init(
            rngs={"params": subkey1},
            x=self.inp_log_fn(self.spatial_data.X[0:1]),
            mode="spatial",
            key=subkey2,
        )["params"]

        lr_sched = optax.exponential_decay(init_lr, decay_steps, 0.5, staircase=True)
        tx = optax.adam(lr_sched)  #

        return TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, metrics=Metrics.empty()
        )

    @partial(jit, static_argnums=(0,))
    def train_step(self, state, spatial_inp, spatial_COVET, sc_inp, key=random.key(0)):
        """
        :meta private:
        """

        key, subkey1, subkey2 = random.split(key, num=3)

        def loss_fn(params):
            spatial_enc_mu, spatial_enc_logstd, spatial_dec_exp, spatial_dec_cov = (
                state.apply_fn(
                    {"params": params},
                    x=self.inp_log_fn(spatial_inp),
                    mode="spatial",
                    key=subkey1,
                )
            )
            sc_enc_mu, sc_enc_logstd, sc_dec_exp = state.apply_fn(
                {"params": params},
                x=self.inp_log_fn(sc_inp[:, : spatial_inp.shape[-1]]),
                mode="sc",
                key=subkey2,
            )

            spatial_exp_like = self.factor_spatial(spatial_inp, spatial_dec_exp)
            sc_exp_like = self.factor_sc(sc_inp, sc_dec_exp)
            spatial_cov_like = jnp.mean(
                AOT_Distance(spatial_COVET, self.grammian_cov(spatial_dec_cov))
            )
            kl_div = jnp.mean(KL(spatial_enc_mu, spatial_enc_logstd)) + jnp.mean(
                KL(sc_enc_mu, sc_enc_logstd)
            )

            loss = (
                -self.spatial_coeff * spatial_exp_like
                - self.sc_coeff * sc_exp_like
                - self.cov_coeff * spatial_cov_like
                + self.kl_coeff * kl_div
            )

            return (
                loss,
                [sc_exp_like, spatial_exp_like, spatial_cov_like, kl_div * 0.5],
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return (state, loss)

    def train(
        self,
        training_steps=16000,
        batch_size=128,
        verbose=16,
        init_lr=0.0001,
        decay_steps=4000,
        key=random.key(0),
    ):
        """
        Set up optimization parameters and train the ENVI moodel


        :param training_steps: (int) number of gradient descent steps to train ENVI (default 16000)
        :param batch_size: (int) size of spatial and single-cell profiles sampled for each training step  (default 128)
        :param verbose: (int) amount of steps between each loss print statement (default 16)
        :param init_lr: (float) initial learning rate for ADAM optimizer with exponential decay (default 1e-4)
        :param decay_steps: (int) number of steps before each learning rate decay (default 4000)
        :param key: (jax.random.key) random seed (default jax.random.key(0))

        :return: nothing
        """

        batch_size = min(
            self.sc_data.shape[0], min(self.spatial_data.shape[0], batch_size)
        )

        key, subkey = random.split(key)
        state = self.create_train_state(
            subkey, init_lr=init_lr, decay_steps=decay_steps
        )
        self.params = state.params

        tq = trange(training_steps, leave=True, desc="")
        sc_loss_mean, spatial_loss_mean, cov_loss_mean, kl_loss_mean, count = (
            0,
            0,
            0,
            0,
            0,
        )

        sc_X = self.sc_data.X
        spatial_X = self.spatial_data.X
        spatial_COVET = self.spatial_data.obsm["COVET_SQRT"]

        for training_step in tq:
            key, subkey1, subkey2 = random.split(key, num=3)

            batch_spatial_ind = random.choice(
                key=subkey1,
                a=self.spatial_data.shape[0],
                shape=[batch_size],
                replace=False,
            )
            batch_sc_ind = random.choice(
                key=subkey2, a=self.sc_data.shape[0], shape=[batch_size], replace=False
            )

            batch_spatial_exp, batch_spatial_cov = (
                spatial_X[batch_spatial_ind],
                spatial_COVET[batch_spatial_ind],
            )
            batch_sc_exp = sc_X[batch_sc_ind]

            key, subkey = random.split(key)

            state, loss = self.train_step(
                state, batch_spatial_exp, batch_spatial_cov, batch_sc_exp, key=subkey
            )

            self.params = state.params

            sc_loss_mean, spatial_loss_mean, cov_loss_mean, kl_loss_mean, count = (
                sc_loss_mean + loss[1][0],
                spatial_loss_mean + loss[1][1],
                cov_loss_mean + loss[1][2],
                kl_loss_mean + loss[1][3],
                count + 1,
            )

            if training_step % verbose == 0:
                print_statement = ""
                for metric, value in zip(
                    ["spatial", "sc", "cov", "kl"],
                    [spatial_loss_mean, sc_loss_mean, cov_loss_mean, kl_loss_mean],
                ):
                    print_statement = (
                        print_statement
                        + " "
                        + metric
                        + ": {:.3e}".format(value / count)
                    )

                sc_loss_mean, spatial_loss_mean, cov_loss_mean, kl_loss_mean, count = (
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                tq.set_description(print_statement)
                tq.refresh()  # to show

        self.latent_rep()

    # @partial(jit, static_argnums=(0,))
    def model_encoder(self, x):
        """
        :meta private:
        """

        return self.model.bind({"params": self.params}).encoder(x)

    # @partial(jit, static_argnums=(0,))
    def model_decoder_exp(self, x):
        """
        :meta private:
        """

        return self.model.bind({"params": self.params}).decoder_exp(x)

    # @partial(jit, static_argnums=(0,))
    def model_decoder_cov(self, x):
        """
        :meta private:
        """
        return self.model.bind({"params": self.params}).decoder_cov(x)

    def encode(self, x, mode="spatial", max_batch=64):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([self.inp_log_fn(x), conf_neurons], axis=-1)

        if x_conf.shape[0] < max_batch:
            print("Encoding")
            enc = jnp.split(self.model_encoder(x_conf), 2, axis=-1)[0]
        else:  # For when the GPU can't pass all point-clouds at once
            num_split = int(x_conf.shape[0] / max_batch) + 1
            x_conf_split = np.array_split(x_conf, num_split)
            enc = np.concatenate(
                [
                    jnp.split(self.model_encoder(x_conf_split[split_ind]), 2, axis=-1)[
                        0
                    ]
                    for split_ind in tqdm(range(num_split), desc="Encoding", leave=True)
                ],
                axis=0,
            )
        return enc

    def decode_exp(self, x, mode="spatial", max_batch=64):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)

        if mode == "spatial":
            if x_conf.shape[0] < max_batch:
                print("Decoding expression")
                dec = self.mean_spatial(self.model_decoder_exp(x_conf))
            else:  # For when the GPU can't pass all point-clouds at once
                num_split = int(x_conf.shape[0] / max_batch) + 1
                x_conf_split = np.array_split(x_conf, num_split)
                dec = np.concatenate(
                    [
                        self.mean_spatial(
                            self.model_decoder_exp(x_conf_split[split_ind])
                        )
                        for split_ind in tqdm(range(num_split), desc="Decoding expression", leave=True)
                    ],
                    axis=0,
                ) 
        else:
            if x_conf.shape[0] < max_batch:
                print("Decoding expression")
                dec = self.mean_sc(
                    self.model.bind({"params": self.params}).decoder_exp(x_conf)
                )
            else:  # For when the GPU can't pass all point-clouds at once
                num_split = int(x_conf.shape[0] / max_batch) + 1
                x_conf_split = np.array_split(x_conf, num_split)
                dec = np.concatenate(
                    [
                        self.mean_sc(self.model_decoder_exp(x_conf_split[split_ind]))
                        for split_ind in tqdm(range(num_split), desc="Decoding expression", leave=True)
                    ],
                    axis=0,
                )
        return dec

    def decode_cov(self, x, max_batch=64):
        """
        :meta private:
        """

        if x.shape[0] < max_batch:
            print("Decoding covet")
            dec = self.grammian_cov(self.model_decoder_cov(x))
        else:  # For when the GPU can't pass all point-clouds at once
            num_split = int(x.shape[0] / max_batch) + 1
            x_split = np.array_split(x, num_split)
            dec = np.concatenate(
                [
                    self.grammian_cov(self.model_decoder_cov(x_split[split_ind]))
                    for split_ind in tqdm(range(num_split), desc="Decoding covet", leave=True)
                ],
                axis=0,
            )
        return dec

    def latent_rep(self):
        """
        Compute latent embeddings for spatial and single cell data, automatically performed after training

        :return: nothing, adds 'envi_latent' self.spatial_data.obsm and self.spatial_data.obsm
        """
        print("Computing latent representations")
        self.spatial_data.obsm["envi_latent"] = np.asarray(self.encode(
            self.spatial_data.X, mode="spatial"
        ))
        self.sc_data.obsm["envi_latent"] = np.asarray(self.encode(
            self.sc_data[:, self.spatial_data.var_names].X, mode="sc"
        ))

    def impute_genes(self):
        """
        Impute full transcriptome for spatial data

        :return: nothing, adds 'imputation' to self.spatial_data.obsm
        """
        print("Imputing missing genes for spatial data")
        self.spatial_data.obsm["imputation"] = pd.DataFrame(
            self.decode_exp(self.spatial_data.obsm["envi_latent"], mode="sc"),
            columns=self.sc_data.var_names,
            index=self.spatial_data.obs_names,
        )


    def infer_niche_covet(self):
        """
        Predict COVET representation for single-cell data

        :return: nothing, adds 'COVET_SQRT' and 'COVET' to self.sc_data.obsm
        """
        print("Infering niche COVET representation for single-cell data")
        self.sc_data.obsm["COVET_SQRT"] = np.asarray(self.decode_cov(
            self.sc_data.obsm["envi_latent"]
        ))
        self.sc_data.obsm["COVET"] = np.matmul(
            self.sc_data.obsm["COVET_SQRT"], self.sc_data.obsm["COVET_SQRT"]
        )

    def infer_niche_celltype(self, cell_type_key="cell_type"):
        """
        Predict cell type abundence based one ENVI-inferred COVET representations

        :param cell_type_key: (string) key in spatial_data.obs where cell types are stored for environment composition (default 'cell_type')

        :return: nothing, adds 'niche_cell_type' to self.sc_data.obsm & self.spatial_data.obsm
        """
        print("Infering cell type niche composition for single cell data")
        self.spatial_data.obsm["cell_type_niche"] = niche_cell_type(
            self.spatial_data,
            self.k_nearest,
            spatial_key=self.spatial_key,
            cell_type_key=cell_type_key,
            batch_key=self.batch_key,
        )

        regression_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5).fit(
            self.spatial_data.obsm["COVET_SQRT"].reshape(
                [self.spatial_data.shape[0], -1]
            ),
            self.spatial_data.obsm["cell_type_niche"],
        )

        sc_cell_type = regression_model.predict(
            self.sc_data.obsm["COVET_SQRT"].reshape([self.sc_data.shape[0], -1])
        )

        self.sc_data.obsm["cell_type_niche"] = pd.DataFrame(
            sc_cell_type,
            index=self.sc_data.obs_names,
            columns=self.spatial_data.obsm["cell_type_niche"].columns,
        )
