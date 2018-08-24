import autograd.numpy as np

def expectedstats(x):
    return 1./(1 + np.exp(-x))

def mean_to_natural(x):
    return np.log(x/(1 - x))

def logZ(natparam):
    return natparam + np.log(1 + np.exp(-natparam))