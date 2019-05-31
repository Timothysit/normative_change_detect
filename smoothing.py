# Smoothing functions / penalties to be used in hmm2_jax.py
import jax.numpy as np
import numpy as onp  # original numpy for indexed assignment/mutation (outside context of differentiation)
from jax import vmap
import jax.random as jaxrandom

def second_derivative_penalty(x, lambda_weight=1):
    """
    Second derivative penalty on the smoothness of a curve
    :param x:
    :param lambda_weight:
    :return:
    """

    # TODO: JAX numpy does not have gradient yet, may need to write that myself...
    x_prime = onp.gradient(x)
    x_pprime = onp.gradient(x_prime)


    return lambda_weight * np.sum(x_pprime ** 2)


def test_penalty_func(penalty_func, scale_list=[0.1, 1, 2]):

    x = np.linspace(1, 1000, 1000)
    for scale in scale_list:
        y_w_noise = np.sin(x) + onp.random.normal(scale=scale, size=len(x))
        smoothness_cost = penalty_func(y_w_noise, lambda_weight=1)
        print("Scale: %.2f" % scale)
        print("Smoothness penalty: %.2f" % smoothness_cost)


def rbf_kernel(xs, lengthscale=1, amplitude=1):
    # From here: https://github.com/google/jax/pull/97
    def gram(kernel, xs):
        return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs))(xs)

    rbf_kernel = lambda x, y: np.exp(- np.sum(x - y) ** 2 / (2 * lengthscale ** 2)) * amplitude

    return gram(rbf_kernel, xs)

def kl_div(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    Source: https://gist.github.com/swayson/86c296aa354a555536e6765bbe726ff7
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def fill_lower_diag(a):
    """
    Fill lower triangle of a matrix using a vector.
    Source: https://stackoverflow.com/questions/51439271/convert-1d-array-to-lower-triangular-matrix
    :param a: 1D numpy array
    :return:
    """
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=int)
    out[mask] = a
    return out

def make_lower_triangle(lower_triangle_params, diag_params):

    lower_triangle = fill_lower_diag(lower_triangle_params)
    # lower_triangle = np.fill_diagonal(diag_params)  # Waiting for this to be implemented in JAX numpy

    # apply sigmoid to diag_params to constrain to positive non-zero values
    lower_triangle = lower_triangle + np.eye(np.shape(lower_triangle)[0]) * standard_sigmoid(diag_params)

    return lower_triangle

def forward_inference_w_elbo(signal):
    key = random.PRNGKey(777)
    epsilon = jaxrandom.normal(key, (len(signal.flatten()),))

    mean_vector = np.zeros((np.shape(epsilon)[0], 1))
    hazard_rate = lower_triangle * epsilon + mean_vector


    sampled_q = 1 # TODO: N(m, L \times L.T)

    # sample from kernel TODO: still need to sort out the kernel
    p_from_kernel = jaxrandom.normal(key, (len(signal.flatten()), len(signal.flatten()))) * rbf_kernel()

    # foward inference stuff here
    # TODO: use the obtained hazard rate to create transition matrix

    posterior = 1

    return np.exp(posterior + kl_div(sampled_q, p_from_kernel))


def standard_sigmoid(input):
    # Naive version:
    # output = 1.0 / (1.0 + np.exp(-input))
    # Numerically stable version
    """
    if input >= 0:
        z = np.exp(-input)
        output = 1.0 / (1.0 + z)
    else:
        z = np.exp(input)
        output = z / (1.0 + z)
    """

    # Numerically stable and applied to an array
    output = np.where(input >= 0,
                    1.0 / (1.0 + np.exp(-input)),
                    np.exp(input) / (1.0 + np.exp(input)))

    return output

def main():
    test_penalty_func(second_derivative_penalty, scale_list=[0.1, 1, 2])
    a = np.eye(3)
    print(a * np.array([0, 4, 99]))
if __name__ == '__main__':
    main()






