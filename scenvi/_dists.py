import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as jnd
import ott
from ott.solvers import linear


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


# from Wasserstein Wormhole
def S2(x, y, eps, lse_mode):
                            
    """
    Calculate Sinkhorn Divergnece (S2) between two weighted point clouds


    :param x: (list) list with two elements, the first (x[0]) being the point-cloud coordinates and the second (x[1]) being each points weight)
    :param y: (list) list with two elements, the first (y[0]) being the point-cloud coordinates and the second (y[1]) being each points weight)
    :param eps: (float) coefficient of entropic regularization
    :param lse_mode: (bool) whether to use log-sum-exp mode (if True, more stable for smaller eps, but slower) or kernel mode (default False)
    
    :return S2: Sinkhorn Divergnece between x and y 
    """ 

    # x,a = x[0], x[1]
    # y,b = y[0], y[1]
    # default to uniform without specifying a and b
        
    ot_solve_xy = linear.solve(
        ott.geometry.pointcloud.PointCloud(x, y, cost_fn=None, epsilon = eps),
        # a = a,
        # b = b,
        lse_mode=lse_mode,
        min_iterations=0,
        max_iterations=100)

    ot_solve_xx = linear.solve(
    ott.geometry.pointcloud.PointCloud(x, x, cost_fn=None, epsilon = eps),
    # a = a,
    # b = a,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)
    
    ot_solve_yy = linear.solve(
    ott.geometry.pointcloud.PointCloud(y, y, cost_fn=None, epsilon = eps),
    # a = b,
    # b = b,
    lse_mode=lse_mode,
    min_iterations=0,
    max_iterations=100)

    return(ot_solve_xy.reg_ot_cost - 0.5 * ot_solve_xx.reg_ot_cost - 0.5 * ot_solve_yy.reg_ot_cost)
