from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.fixed_points import fixed_point
from svae.distributions import gamma, gaussian
from svae.util import flat, unbox

def _global_kl(global_prior_natparams, global_natparams):
    stats = flat(gamma.expectedstats(global_natparams))
    natparam_difference = flat(global_natparams) - flat(global_prior_natparams)
    return np.dot(natparam_difference, stats) - (gamma.logZ(global_natparams) - gamma.logZ(global_prior_natparams))

def _local_kl(global_stats, local_natparams, local_stats):
    global_natparams = gaussian.pack_dense(np.array([[-.5*global_stats[1]]]), np.array([0]), 0, .5*global_stats[0])
    return np.tensordot(local_natparams - global_natparams, local_stats, 4) - gaussian.logZ(local_natparams)

def _local_ep_update(global_stats, encoder_potentials, (fwd_messages, bwd_messages)):
    N, T, C, _ = encoder_potentials[0].shape
    encoder_natparams = gaussian.pack_dense(*encoder_potentials[:2])
    p = encoder_potentials[2]

    for t in range(T-1):
        cavity_natparams = fwd_messages[:,t] + bwd_messages[:,t] # (N,3,3)
        r = p[:,t] * np.exp(gaussian.logZ(encoder_natparams[:,t] + cavity_natparams) - gaussian.logZ(encoder_natparams[:,t])) # (N,C)
        stats = np.sum(r[...,None,None]*gaussian.expectedstats(encoder_natparams[:,t] + cavity_natparams[:,None]), axis=1) # (N,3,3)
        f_natparams = gaussian.mean_to_natural(stats) - cavity_natparams # (N,3,3)
        m, v = gaussian.natural_to_standard(
            f_natparams + fwd_messages[:,t] + gaussian.pack_dense(-.5*np.reshape(global_stats[0], (1,1)), np.zeros(1)))
        fwd_messages[:,t+1] = gaussian.mean_to_natural(gaussian.pack_dense(v + np.pow(m, 2), m)) # (N,3,3)

    for t in reversed(xrange(T, 0, -1)):
        cavity_natparams = fwd_messages[:,t] + bwd_messages[:,t] # (N,3,3)
        r = p[:,t] * np.exp(gaussian.logZ(encoder_natparams[:,t] + cavity_natparams) - gaussian.logZ(encoder_natparams[:,t])) # (N,C)
        stats = np.sum(r[...,None,None]*gaussian.expectedstats(encoder_natparams[:,t] + cavity_natparams[:,None]), axis=1) # (N,3,3)
        f_natparams = gaussian.mean_to_natural(stats) - cavity_natparams # (N,3,3)
        m, v = gaussian.natural_to_standard(
            f_natparams + bwd_messages[:,t] + gaussian.pack_dense(-.5*np.reshape(global_stats[0], (1,1)), np.zeros(1)))
        bwd_messages[:,t-1] = gaussian.mean_to_natural(gaussian.pack_dense(v + np.pow(m, 2), m)) # (N,3,3)

    return fwd_messages, bwd_messages

def local_inference(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    N, T, C, D = encoder_potentials[0].shape
    assert(D == 1)

    def make_fpfun((global_stats, encoder_potentials)):
        return lambda x: \
            _local_ep_update(global_stats, encoder_potentials, x)

    def diff(a, b):
        return np.sum(np.abs(a[0]- b[0]))

    def init_x0():
        fwd_messages = np.stack([
            gaussian.pack_dense(-.5*global_stats[1]*np.ones((N,1,1,1)), np.zeros((N,1,1)), 0, 0),
            gaussian.pack_dense(-.01*np.ones((N, T, 1, 1)), .01*npr.randn(N, T, 1)], axis=1)
        bwd_messages = np.stack([
            gaussian.pack_dense(-.01*np.ones((N, T, 1, 1)), .01*npr.randn(N, T, 1)),
            np.zeros(N,1,3,3)], axis=1)
        return (fwd_messages, bwd_messages)

    _, local_natparams, local_stats = fixed_point(make_fpfun, (global_stats, encoder_potentials), init_x0(), diff, tol=1e-3)

    local_samples = gaussian.natural_sample(local_natparams[1], n_samples)
    global_stats = () # add gamma stats

    # niw_stats = np.tensordot(z_pairstats, local_stats[1], [(0,1), (0,1)])
    # beta_stats = np.sum(local_stats[0], axis=0), np.sum(1-local_stats[0], axis=0)

    local_kl = _local_kl(unbox(global_stats), local_natparams, local_stats)
    global_kl = _global_kl(global_prior_natparams, global_natparams)
    return local_samples, local_natparams, unbox(local_stats), global_kl, local_kl

def pgm_expectedstats(natparams):
    return gamma.expectedstats(natparams)