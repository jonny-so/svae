from svae.models.dlds import init_pgm_natparams, local_inference, make_loglike, pgm_expectedstats
from svae.nnet import init_layer_glorot, init_mlp, init_gresnet_mix
from svae.optimizers import adam
from svae.svae import make_gradfun
from toolz import identity
import autograd.numpy as np
import autograd.numpy.random as npr
import scipy.stats

def gen_data(N, T, tau, scale):
    x = np.cumsum(npr.randn(N,T)*np.sqrt(1./tau), axis=1)[...,None]
    return np.array(npr.rand(*x.shape) > scipy.stats.norm.cdf(x, scale=scale), dtype=np.int32)

if __name__ == "__main__":
    # use constant data set
    npr.seed(0)

    N = 1000
    T = 100
    C = 1
    tau = 1/(1.0/100)
    y = gen_data(N, T, tau, scale=1.0)

    # global prior and variational posterior natparams
    global_prior_natparams = init_pgm_natparams(tau, np.power(10.0, 2))
    global_natparams = init_pgm_natparams(tau, np.power(20.0, 2))

    # construct recognition and decoder networks and initialize them
    encoder, phi = \
        init_gresnet_mix(1, 1, C, [
            (40, np.tanh, init_layer_glorot),
            (40, np.tanh, init_layer_glorot)])
    decoder, gamma = \
        init_mlp(1, [
            (40, np.tanh, init_layer_glorot),
            (40, np.tanh, init_layer_glorot),
            (1, identity, init_layer_glorot)])
    loglike = make_loglike(decoder)

    params = global_natparams, gamma, phi

    def callback(i, val, params, grad):
        print('{}: {}'.format(i, val))

    np.seterr(all='raise')

    gradfun = make_gradfun(
        local_inference, encoder, loglike, global_prior_natparams, pgm_expectedstats, y)

    params = adam(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e4, callback=callback),
                  params, num_iters=1000, step_size=1e-3)