import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as jnd


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

    log_prob = jnd.Poisson(rate=l).log_prob(sample)
    return jnp.mean(log_prob, axis=-1)


def log_nb_pdf(sample, r, p):
    """
    :meta private:
    """

    log_prob = jnd.NegativeBinomial(total_count=r, logits=p).log_prob(sample)
    return jnp.mean(log_prob, axis=-1)


def log_zinb_pdf(sample, r, p, d):
    """
    :meta private:
    """

    log_prob = jnd.Inflated(
        jnd.NegativeBinomial(total_count=r, logits=p), inflated_loc_logits=d
    ).log_prob(sample)
    return jnp.mean(log_prob, axis=-1)


def log_normal_pdf(sample, mean):
    """
    :meta private:
    """

    log_prob = jnd.Normal(loc=mean, scale=1).log_prob(sample)
    return jnp.mean(log_prob, axis=-1)


def AOT_Distance(sample, mean):
    """
    :meta private:
    """

    sample = jnp.reshape(sample, [sample.shape[0], -1])
    mean = jnp.reshape(mean, [mean.shape[0], -1])
    log_prob = -jnp.square(sample - mean)
    return jnp.mean(log_prob, axis=-1)
