import autograd.numpy as np
import autograd.scipy as scipy
from svae.util import log1pexp

def expectedstats(x):
    return scipy.special.expit(x)

def mean_to_natural(x):
    return scipy.special.logit(x)

def logZ(natparam):
    return np.where(natparam > 0, natparam + log1pexp(-natparam), log1pexp(natparam))