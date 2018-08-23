from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import betaln, digamma, gammaln

def natural_to_mean(a, b):
    alpha = a + 1
    beta = b + 1
    return digamma(alpha) - digamma(alpha + beta), digamma(beta) - digamma(alpha + beta)

def logZ(a, b):
    alpha = a + 1
    beta = b + 1
    return betaln(alpha, beta)