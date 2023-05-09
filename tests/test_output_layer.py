import numpy as np
import tensorflow as tf

from scENVI.output_layer import LinearLayer, ConstantLayer, ENVIOutputLayer

# Test data
input_dim = 5
units = 3
batch_size = 2
input_data = np.random.randn(batch_size, input_dim).astype(np.float32)

# Test initializers
kernel_init = tf.keras.initializers.RandomNormal()
bias_init = tf.keras.initializers.Zeros()


# Test LinearLayer
def test_linear_layer():
    linear_layer = LinearLayer(units, input_dim, kernel_init, bias_init, "linear_test")
    output = linear_layer(input_data)
    assert output.shape == (
        batch_size,
        units,
    ), f"Expected output shape ({batch_size}, {units}), but got {output.shape}"


# Test ConstantLayer
def test_constant_layer():
    constant_layer = ConstantLayer(units, input_dim, bias_init, "constant_test")
    output = constant_layer(input_data)
    assert output.shape == (
        batch_size,
        units,
    ), f"Expected output shape ({batch_size}, {units}), but got {output.shape}"


# Test ENVIOutputLayer
def test_envi_output_layer():
    layer = ENVIOutputLayer(
        input_dim, units, kernel_init, bias_init, spatial_dist="pois", sc_dist="nb"
    )
    r = layer(input_data)
    assert r.shape == (
        batch_size,
        units,
    ), f"Expected output shape ({batch_size}, {units}), but got {r.shape}"


def test_negative_binomial_sc_envi_layer():
    layer = ENVIOutputLayer(
        input_dim, units, kernel_init, bias_init, spatial_dist="pois", sc_dist="nb"
    )
    r, p = layer(input_data, mode="sc")
    shape = (batch_size, units)
    assert r.shape == shape, f"Expected output shape {shape}, but got {r.shape}"
    assert p.shape == shape, f"Expected output shape {shape}, but got {p.shape}"


def test_zinb_sc_envi_layer():
    layer = ENVIOutputLayer(
        input_dim, units, kernel_init, bias_init, spatial_dist="pois", sc_dist="zinb"
    )
    r, p, d = layer(input_data, mode="sc")
    shape = (batch_size, units)
    assert r.shape == shape, f"Expected output shape {shape}, but got {r.shape}"
    assert p.shape == shape, f"Expected output shape {shape}, but got {p.shape}"
    assert d.shape == shape, f"Expected output shape {shape}, but got {d.shape}"


def test_share_disp_envi_layer():
    layer = ENVIOutputLayer(
        input_dim,
        units,
        kernel_init,
        bias_init,
        spatial_dist="nb",
        sc_dist="nb",
        share_disp=True,
        const_disp=False,
    )
    shape = (batch_size, units)
    r, p = layer(input_data, mode="spatial")
    assert r.shape == shape, f"Expected shape {shape}, but got {r.shape}"
    assert p.shape == shape, f"Expected shape {shape}, but got {p.shape}"
    sc_r, sc_p = layer(input_data, mode="sc")
    assert sc_r.shape == shape, f"Expected output shape {shape}, but got {sc_r.shape}"
    assert sc_p.shape == shape, f"Expected output shape {shape}, but got {sc_p.shape}"

    assert np.array_equal(p, sc_p), "Spatial and single cell dispersion should be equal"


def test_share_disp_zinb_envi_layer():
    layer = ENVIOutputLayer(
        input_dim,
        units,
        kernel_init,
        bias_init,
        spatial_dist="zinb",
        sc_dist="nb",
        share_disp=True,
        const_disp=False,
    )
    shape = (batch_size, units)
    r, p, d = layer(input_data, mode="spatial")
    assert r.shape == shape, f"Expected shape {shape}, but got {r.shape}"
    assert p.shape == shape, f"Expected shape {shape}, but got {p.shape}"
    assert d.shape == shape, f"Expected shape {shape}, but got {d.shape}"
    sc_r, sc_p = layer(input_data, mode="sc")
    assert sc_r.shape == shape, f"Expected output shape {shape}, but got {sc_r.shape}"
    assert sc_p.shape == shape, f"Expected output shape {shape}, but got {sc_p.shape}"

    assert np.array_equal(p, sc_p), "Spatial and single cell dispersion should be equal"
