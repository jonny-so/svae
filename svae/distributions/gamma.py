import autograd.numpy as np
from autograd.scipy.special import gammaln, digamma

def expectedstats(natparam):
    alpha = natparam[...,0] + 1
    beta = -natparam[...,1]
    assert(alpha > 0)
    assert(beta > 0)
    return np.stack([digamma(alpha) - np.log(beta), alpha/beta], axis=-1)

def logZ(natparam):
    alpha = natparam[...,0] + 1
    beta = -natparam[...,1]
    assert(alpha > 0)
    assert(beta > 0)
    return gammaln(alpha) - alpha*np.log(beta)