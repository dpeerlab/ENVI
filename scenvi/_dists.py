"""Log-densities and matrix helpers for ENVI's decoders.

These were tensorflow_probability's jax substrate. tfp's last release is 0.25.0
(November 2024) and it is unmaintained, while its pinned version here (<0.23)
fails to import against any recent jax:

    AttributeError: module 'jax.interpreters.xla' has no attribute
    'pytype_aval_mappings'

The surface actually used was five functions, all of them closed-form, so they
are written out here against jax directly. Each one is checked against the tfp
original in tests/test_dists.py.
"""

import math

import jax.numpy as jnp
from jax.nn import log_sigmoid
from jax.scipy.special import gammaln, xlogy

#: ``log(2 * pi)``, the Gaussian normalization constant.
LOG_TWO_PI = math.log(2.0 * math.pi)


def KL(mean, log_std):
    """
    :meta private:
    """
    KL = 0.5 * (jnp.square(mean) + jnp.square(jnp.exp(log_std)) - 2 * log_std)
    return jnp.mean(KL, axis=-1)


def log_pos_pdf(sample, l):  # noqa: E741
    """
    :meta private:
    """

    # Poisson(rate=l): k log(l) - l - log(k!). xlogy keeps k = 0, l = 0 at 0
    # rather than nan, which is what tfp does.
    log_prob = xlogy(sample, l) - l - gammaln(sample + 1.0)
    return jnp.mean(log_prob, axis=-1)


def log_nb_pdf(sample, r, p):
    """
    :meta private:
    """

    # NegativeBinomial(total_count=r, logits=p), i.e. p is the logit of the
    # per-trial success probability:
    #   log C(k + r - 1, k) + k log(sigmoid(p)) + r log(1 - sigmoid(p))
    binomial_coefficient = gammaln(sample + r) - gammaln(sample + 1.0) - gammaln(r)
    log_prob = binomial_coefficient + sample * log_sigmoid(p) + r * log_sigmoid(-p)
    return jnp.mean(log_prob, axis=-1)


def log_zinb_pdf(sample, r, p, d):
    """
    :meta private:
    """

    # Zero-inflated negative binomial: with probability sigmoid(2d) the sample is
    # a structural zero, otherwise it is drawn from the negative binomial. Only
    # the k = 0 branch mixes the two, and it is summed in log space.
    #
    # The factor of two is not a typo. tfp built the mixture's categorical from
    # logits [d, -d], whose difference is 2d, so the inflation weight is
    # sigmoid(2d) and not sigmoid(d). Getting this wrong rescales the
    # zero-inflation parameter silently -- see tests/test_dists.py.
    binomial_coefficient = gammaln(sample + r) - gammaln(sample + 1.0) - gammaln(r)
    log_nb = binomial_coefficient + sample * log_sigmoid(p) + r * log_sigmoid(-p)

    log_nb_at_zero = r * log_sigmoid(-p)
    log_prob = jnp.where(
        sample == 0,
        jnp.logaddexp(log_sigmoid(2.0 * d), log_sigmoid(-2.0 * d) + log_nb_at_zero),
        log_sigmoid(-2.0 * d) + log_nb,
    )
    return jnp.mean(log_prob, axis=-1)


def log_normal_pdf(sample, mean):
    """
    :meta private:
    """

    log_prob = -0.5 * (jnp.square(sample - mean) + LOG_TWO_PI)
    return jnp.mean(log_prob, axis=-1)


def AOT_Distance(sample, mean):
    """
    :meta private:
    """

    sample = jnp.reshape(sample, [sample.shape[0], -1])
    mean = jnp.reshape(mean, [mean.shape[0], -1])
    log_prob = -jnp.square(sample - mean)
    return jnp.mean(log_prob, axis=-1)


def fill_triangular(x):
    """Pack ``x`` into the lower triangle of a square matrix, tfp's way.

    Reproduces ``tfp.math.fill_triangular``, whose fill order is not the obvious
    one -- the last ``m - n`` entries are laid down first, then the whole vector
    reversed on top, and the lower triangle taken::

        [1, 2, 3, 4, 5, 6] -> [[4, 0, 0],
                               [6, 5, 0],
                               [3, 2, 1]]

    :param x: (array) ``(..., n * (n + 1) / 2)`` values to pack

    :return: (array) ``(..., n, n)`` lower-triangular matrices

    :meta private:
    """

    m = x.shape[-1]
    # m = n (n + 1) / 2, so n is the positive root of n^2 + n - 2m.
    n = int(round((math.sqrt(1.0 + 8.0 * m) - 1.0) / 2.0))
    if n * (n + 1) // 2 != m:
        raise ValueError(f"last dimension {m} is not a triangular number")

    packed = jnp.concatenate([x[..., n:], jnp.flip(x, axis=-1)], axis=-1)
    return jnp.tril(jnp.reshape(packed, (*x.shape[:-1], n, n)))
