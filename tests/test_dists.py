"""The tfp-free log-densities must agree with the tensorflow_probability originals.

scenvi no longer depends on tensorflow_probability, so these run only where it is
still installed and importable -- which, given the <0.23 pin fails against any
recent jax, in practice means a deliberately pinned environment. They are the
record that the replacements in `scenvi/_dists.py` reproduce what they replaced.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
tfp = pytest.importorskip("tensorflow_probability.substrates.jax")
jnd = tfp.distributions

from scenvi._dists import (  # noqa: E402
    fill_triangular,
    log_nb_pdf,
    log_normal_pdf,
    log_pos_pdf,
    log_zinb_pdf,
)

SHAPE = (16, 32)


@pytest.fixture(autouse=True)
def double_precision():
    """Compare the formulae, not float32 rounding, whatever the ambient config."""
    from jax.experimental import enable_x64

    with enable_x64():
        yield


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def counts(rng):
    """Non-negative counts, deliberately including exact zeros."""
    sample = rng.poisson(2.0, size=SHAPE).astype(np.float64)
    assert (sample == 0).any(), "the zero branch must be exercised"
    return jnp.asarray(sample)


def test_poisson_matches_tfp(rng, counts):
    rate = jnp.asarray(rng.uniform(0.05, 10.0, size=SHAPE))

    expected = jnp.mean(jnd.Poisson(rate=rate).log_prob(counts), axis=-1)
    np.testing.assert_allclose(log_pos_pdf(counts, rate), expected, rtol=1e-12, atol=1e-12)


def test_negative_binomial_matches_tfp(rng, counts):
    total_count = jnp.asarray(rng.uniform(0.5, 20.0, size=SHAPE))
    logits = jnp.asarray(rng.normal(0.0, 3.0, size=SHAPE))

    expected = jnp.mean(
        jnd.NegativeBinomial(total_count=total_count, logits=logits).log_prob(counts), axis=-1
    )
    np.testing.assert_allclose(
        log_nb_pdf(counts, total_count, logits), expected, rtol=1e-11, atol=1e-11
    )


def test_zero_inflated_negative_binomial_matches_tfp(rng, counts):
    total_count = jnp.asarray(rng.uniform(0.5, 20.0, size=SHAPE))
    logits = jnp.asarray(rng.normal(0.0, 3.0, size=SHAPE))
    inflation = jnp.asarray(rng.normal(0.0, 3.0, size=SHAPE))

    expected = jnp.mean(
        jnd.Inflated(
            jnd.NegativeBinomial(total_count=total_count, logits=logits),
            inflated_loc_logits=inflation,
        ).log_prob(counts),
        axis=-1,
    )
    np.testing.assert_allclose(
        log_zinb_pdf(counts, total_count, logits, inflation), expected, rtol=1e-11, atol=1e-11
    )


def test_normal_matches_tfp(rng):
    sample = jnp.asarray(rng.normal(size=SHAPE))
    mean = jnp.asarray(rng.normal(size=SHAPE))

    expected = jnp.mean(jnd.Normal(loc=mean, scale=1).log_prob(sample), axis=-1)
    np.testing.assert_allclose(log_normal_pdf(sample, mean), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16])
def test_fill_triangular_matches_tfp(rng, n):
    x = jnp.asarray(rng.normal(size=(5, n * (n + 1) // 2)))

    np.testing.assert_array_equal(fill_triangular(x), tfp.math.fill_triangular(x))


def test_extreme_parameters_match_tfp(rng):
    """Saturated logits and large counts, where a naive log(sigmoid(p)) underflows."""
    sample = jnp.asarray(rng.integers(0, 500, size=SHAPE).astype(np.float64))
    total_count = jnp.asarray(rng.uniform(1.0, 500.0, size=SHAPE))
    logits = jnp.asarray(rng.uniform(-40.0, 40.0, size=SHAPE))
    inflation = jnp.asarray(rng.uniform(-40.0, 40.0, size=SHAPE))

    nb = jnd.NegativeBinomial(total_count=total_count, logits=logits)
    np.testing.assert_allclose(
        log_nb_pdf(sample, total_count, logits),
        jnp.mean(nb.log_prob(sample), axis=-1),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        log_zinb_pdf(sample, total_count, logits, inflation),
        jnp.mean(
            jnd.Inflated(nb, inflated_loc_logits=inflation).log_prob(sample), axis=-1
        ),
        rtol=1e-11,
        atol=1e-11,
    )
