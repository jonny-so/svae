import autograd.numpy as np
from autograd.scipy.special import gammaln, digamma

def expectedstats(natparam):
    alpha = natparam[0] + 1
    beta = -natparam[1]
    return digamma(alpha), alpha/beta

def logZ(natparam):
    alpha = natparam[0] + 1
    beta = -natparam[1]
    return gammaln(alpha) - alpha*np.log(beta)