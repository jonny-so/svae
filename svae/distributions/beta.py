from __future__ import division
from autograd.scipy.special import betaln, digamma

def expectedstats(a, b):
    alpha = a + 1
    beta = b + 1
    return digamma(alpha) - digamma(alpha + beta), digamma(beta) - digamma(alpha + beta)

def logZ(a, b):
    alpha = a + 1
    beta = b + 1
    return betaln(alpha, beta)