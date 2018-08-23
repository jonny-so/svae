import autograd.numpy as np
from autograd import hessian

def natural_to_mean(x):
    return 1./(1 + np.exp(-x))

def mean_to_natural(x):
    return np.log(x/(1 - x))

def logZ(natparam):
    return natparam + np.log(1 + np.exp(-natparam))

def natgrad(natparam, g):
    H = hessian(logZ)(natparam)
    return (1./H)*g