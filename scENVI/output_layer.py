import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    """
    Custom keras linear layer

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
    """

    def __init__(self, units, input_dim, kernel_init, bias_init, name):
        super(LinearLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer=kernel_init,
            trainable=True,
            name=name + "/kernel",
        )
        self.b = self.add_weight(
            shape=(units,), initializer=bias_init, trainable=True, name=name + "/bias"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class ConstantLayer(tf.keras.layers.Layer):
    """
    Custom keras constant layer, biases only

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        bias_init (keras initializer): initializer of neural biases
        comm_disp (bool): if True, spatial_dist and sc_dist share dispersion
            parameter(s)
        const_disp (bool): if True, dispertion parameter(s) are only per gene,
            rather there per gene per sample
    """

    def __init__(self, units, input_dim, bias_init, name):
        super(ConstantLayer, self).__init__()
        self.b = self.add_weight(
            shape=(units,), initializer=bias_init, trainable=True, name=name + "/bias"
        )

    def call(self, inputs):
        return tf.tile(self.b[None, :], [inputs.shape[0], 1])


class ENVIOutputLayer(tf.keras.layers.Layer):
    """
    Custom keras layer for ENVI expression decoder output

    Args:
        units (int): number of neurons in the layer
        input_dim (int): dimension of input to layer
        kernel_init (keras initializer): initializer for neural weights
        bias_init (keras initializer): initializer of neural biases
        spatial_dist (str): distribution used to describe spatial data
            (default pois, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
        sc_dist (str): distribution used to describe single cell data
            (default nb, could be 'pois', 'nb', 'zinb', 'norm' or 'full_norm')
    """

    def __init__(
        self,
        input_dim,
        units,
        kernel_init,
        bias_init,
        spatial_dist="pois",
        sc_dist="nb",
        comm_disp=False,
        const_disp=False,
        name="dec_exp_output",
    ):
        super(ENVIOutputLayer, self).__init__()

        self.input_dim = input_dim
        self.units = units

        self.spatial_dist = spatial_dist
        self.sc_dist = sc_dist
        self.comm_disp = comm_disp
        self.const_disp = const_disp

        self.kernel_init = kernel_init
        self.bias_init = bias_init

        self.r = LinearLayer(units, input_dim, kernel_init, bias_init, name=name + "_r")

        if self.comm_disp:
            if self.spatial_dist == "zinb":
                self.p_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

                self.d_spatial = (
                    ConstantLayer(units, input_dim, bias_init + "_d_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_d_spatial",
                    )
                )

            elif self.spatial_dist == "nb" or self.spatial_dist == "full_norm":
                self.p_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

            if self.sc_dist == "zinb":
                self.p_sc = (
                    self.p_spatial
                    if (
                        self.spatial_dist == "zinb"
                        or self.spatial_dist == "nb"
                        or self.spatial_dist == "full_norm"
                    )
                    else (
                        ConstantLayer(units, input_dim, bias_init, name=name + "_p_sc")
                        if self.const_disp
                        else LinearLayer(
                            units,
                            input_dim,
                            kernel_init,
                            bias_init,
                            name=name + "_p_sc",
                        )
                    )
                )

                self.d_sc = (
                    self.d_spatial
                    if (self.spatial_dist == "zinb")
                    else (
                        ConstantLayer(units, input_dim, bias_init, name=name + "_d_sc")
                        if self.const_disp
                        else LinearLayer(
                            units,
                            input_dim,
                            kernel_init,
                            bias_init,
                            name=name + "_d_sc",
                        )
                    )
                )

            elif self.sc_dist == "nb" or self.sc_dist == "full_norm":
                self.p_sc = (
                    self.p_spatial
                    if (
                        self.spatial_dist == "zinb"
                        or self.spatial_dist == "nb"
                        or self.spatial_dist == "full_norm"
                    )
                    else (
                        ConstantLayer(units, input_dim, bias_init, name=name + "_p_sc")
                        if self.const_disp
                        else LinearLayer(
                            units,
                            input_dim,
                            kernel_init,
                            bias_init,
                            name=name + "_r_sc",
                        )
                    )
                )

            if self.spatial_dist == "zinb" or self.sc_dist == "zinb":
                self.p_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

                self.p_sc = self.p_spatial

                self.d_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_d_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_d_spatial",
                    )
                )

                self.d_sc = self.d_spatial

            elif (
                self.spatial_dist == "nb"
                or self.sc_dist == "nb"
                or self.spatial_dist == "full_norm"
                or self.sc_dist == "full_norm"
            ):
                self.p_spatial = (
                    ConstantLayer(
                        units, input_dim, kernel_init, name=name + "_p_spatial"
                    )
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

                self.p_sc = self.p_spatial

        else:
            if self.spatial_dist == "zinb":
                self.p_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

                self.d_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_d_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_d_spatial",
                    )
                )

            elif self.spatial_dist == "nb" or self.spatial_dist == "full_norm":
                self.p_spatial = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_spatial")
                    if self.const_disp
                    else LinearLayer(
                        units,
                        input_dim,
                        kernel_init,
                        bias_init,
                        name=name + "_p_spatial",
                    )
                )

            if self.sc_dist == "zinb":
                self.p_sc = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_sc")
                    if self.const_disp
                    else LinearLayer(
                        units, input_dim, kernel_init, bias_init, name=name + "_p_sc"
                    )
                )

                self.d_sc = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_d_sc")
                    if self.const_disp
                    else LinearLayer(
                        units, input_dim, kernel_init, bias_init, name=name + "_d_sc"
                    )
                )

            elif self.sc_dist == "nb" or self.sc_dist == "full_norm":
                self.p_sc = (
                    ConstantLayer(units, input_dim, bias_init, name=name + "_p_sc")
                    if self.const_disp
                    else LinearLayer(
                        units, input_dim, kernel_init, bias_init, name=name + "_p_sc"
                    )
                )

    def call(self, inputs, mode="spatial"):
        r = self.r(inputs)

        if getattr(self, mode + "_dist") == "zinb":
            p = getattr(self, "p_" + mode)(inputs)
            d = getattr(self, "d_" + mode)(inputs)
            return (r, p, d)

        if (
            getattr(self, mode + "_dist") == "nb"
            or getattr(self, mode + "_dist") == "full_norm"
        ):
            p = getattr(self, "p_" + mode)(inputs)
            return (r, p)

        return r
