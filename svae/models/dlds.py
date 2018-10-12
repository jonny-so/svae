from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.fixed_points import fixed_point
from svae.distributions import gamma, gaussian, univariate_gaussian
from svae.util import expand_diagonal, flat, replace, softmax, mvp, unbox, sigmoid

def _local_samples(global_stats, local_messages, num_samples):
    fwd_messages, bwd_messages, obs_messages = local_messages
    N, T = fwd_messages.shape[:2]
    samples = [
        univariate_gaussian.natural_sample(fwd_messages[:,0] + bwd_messages[:,0] + obs_messages[:,0], num_samples)]
    for t in xrange(1, T):
        in_message = univariate_gaussian.standard_to_natural(np.ones((num_samples,N))/global_stats[1], samples[-1])
        samples.append(np.squeeze(
            univariate_gaussian.natural_sample(in_message + bwd_messages[:, t] + obs_messages[:, t], 1), 0))
    return np.stack(samples, axis=-1)[...,None]

def _dense_local_samples(global_stats, local_messages, num_samples):
    return np.swapaxes(gaussian.natural_sample(
            _dense_local_natparams(global_stats, local_messages), num_samples)[...,None], 1, 2)

def _global_kl(global_prior_natparams, global_natparams):
    stats = gamma.expectedstats(global_natparams)
    natparam_difference = global_natparams - global_prior_natparams
    return np.dot(natparam_difference, stats) - (gamma.logZ(global_natparams) - gamma.logZ(global_prior_natparams))

def _local_logZ(global_stats, local_messages):
    fwd_messages, _, obs_messages = local_messages
    N, T = fwd_messages.shape[:2]
    tau = np.pad(np.repeat(global_stats[1], T-1), (0,1), 'constant')
    res = np.zeros(N)
    v, m = univariate_gaussian.natural_to_standard(fwd_messages + obs_messages)
    j = 1./v
    for t in range(T):
        res -= .5*np.log(tau[t] + j[:,t])
        res += .5*np.power(m[:,t], 2) * j[:,t] / (1 + v[:,t]*tau[t])
    return np.sum(res)

def _dense_local_logZ(global_stats, local_messages):
    return gaussian.logZ(_dense_local_natparams(global_stats, local_messages))

def _dense_local_natparams(global_stats, local_messages):
    fwd_messages, bwd_messages, obs_messages = local_messages
    N, T = fwd_messages.shape[:2]
    neghalfj = univariate_gaussian.unpack_dense(obs_messages)[0] - global_stats[1] \
        + np.pad(.5*global_stats[1]*np.ones((N,1)), ((0,0),(T-1,0)), 'constant')
    m = univariate_gaussian.natural_to_standard(fwd_messages + bwd_messages + obs_messages)[1]
    A = expand_diagonal(neghalfj) \
        +.5*np.tile(np.diag(global_stats[1]*np.ones(T-1), 1), (N,1,1)) \
        +.5*np.tile(np.diag(global_stats[1]*np.ones(T-1), -1), (N,1,1))
    b = mvp(-2*A, m)
    return gaussian.pack_dense(A, b)

# correct up to a constant, but this means KL is not necessarily >= 0
def _prior_local_logZ(global_stats, N, T):
    return -N*T*.5*global_stats[0]

def _dense_prior_local_logZ(global_stats, N, T):
    neghalfj = np.pad(.5*global_stats[1]*np.ones((N,1)), ((0,0),(T-1,0)), 'constant') - global_stats[1]
    A = expand_diagonal(neghalfj) \
        +.5*np.tile(np.diag(global_stats[1]*np.ones(T-1), 1), (N,1,1)) \
        +.5*np.tile(np.diag(global_stats[1]*np.ones(T-1), -1), (N,1,1))
    b = np.zeros((N,T))
    return gaussian.logZ(gaussian.pack_dense(A, b))

def _local_kl(global_stats, local_messages, local_stats):
    _, _, obs_messages = local_messages
    singleton_stats, _ = local_stats
    # print('{} {}'.format(_local_logZ(global_stats, local_messages), _dense_local_logZ(global_stats, local_messages)))
    return np.tensordot(obs_messages, singleton_stats, 3) - (
        _dense_local_logZ(global_stats, local_messages) - _dense_prior_local_logZ(global_stats, *obs_messages.shape[:2]))

def _local_stats(global_stats, local_messages):
    fwd_messages, bwd_messages, obs_messages = local_messages
    N, T = fwd_messages.shape[:2]
    tau = global_stats[1]
    singleton_stats = univariate_gaussian.expectedstats(fwd_messages + bwd_messages + obs_messages)
    pairwise_natparams = gaussian.pack_dense(
        np.stack((
            fwd_messages[:,:-1,0] + obs_messages[:,:-1,0] - .5*tau,
            .5*np.tile(tau, (N,T-1)),
            .5*np.tile(tau, (N,T-1)),
            bwd_messages[:,1:,0] + obs_messages[:,1:,0] - .5*tau), axis=-1).reshape(N,T-1,2,2),
        np.stack((
            fwd_messages[:, :-1, 1] + obs_messages[:, :-1, 1],
            bwd_messages[:, 1:, 1] + obs_messages[:, 1:, 1]), axis=-1).reshape(N,T-1,2))
    pairwise_stats = gaussian.unpack_dense(gaussian.expectedstats(pairwise_natparams))[0][...,0,1]
    return singleton_stats, pairwise_stats

def _local_ep_update(global_stats, encoder_potentials, (fwd_messages, bwd_messages, obs_messages)):

    N, T, C = encoder_potentials[0].shape
    encoder_natparams = univariate_gaussian.pack_dense(*encoder_potentials[:2]) # (N,T,C,2)
    encoder_logits = encoder_potentials[2] # (N,T,C)

    def propagate(in_messages, t):
        cavity_natparams = fwd_messages[:,t] + bwd_messages[:,t] # (N,2)
        r = softmax(encoder_logits[:,t] +
            univariate_gaussian.logZ(encoder_natparams[:,t] + cavity_natparams[:,None]) -
            univariate_gaussian.logZ(encoder_natparams[:,t])) # (N,C)
        marginal_stats = np.sum(r[...,None]*univariate_gaussian.expectedstats(
            encoder_natparams[:,t] + cavity_natparams[:,None]), axis=1) # (N,2)
        obs_natparams = univariate_gaussian.mean_to_natural(marginal_stats) - cavity_natparams # (N,2)
        out_v = univariate_gaussian.natural_to_standard(obs_natparams + in_messages[:,t])[0] + 1./global_stats[1]
        out_m = univariate_gaussian.natural_to_standard(obs_natparams + in_messages[:,t])[1]
        return obs_natparams, univariate_gaussian.standard_to_natural(out_v, out_m)

    for t in range(T-1):
        obs_message, fwd_message = propagate(fwd_messages, t)
        fwd_messages = replace(fwd_messages, fwd_message, t+1, axis=1)
        obs_messages = replace(obs_messages, obs_message, t, axis=1)
    for t in range(T-1, 0, -1):
        obs_message, bwd_message = propagate(bwd_messages, t)
        bwd_messages = replace(bwd_messages, bwd_message, t-1, axis=1)
        obs_messages = replace(obs_messages, obs_message, t, axis=1)

    return fwd_messages, bwd_messages, obs_messages

def _global_potentials(local_stats):
    N, T, _ = local_stats[0].shape
    a = .5 * N * T
    b = np.sum(local_stats[1]) - np.sum(local_stats[0][..., 0]) + .5*np.sum(local_stats[0][:, -1, 0])
    assert(b < 0)
    return np.array([a, b])

def local_inference_messages(global_stats, encoder_potentials):

    encoder_potentials = \
        np.squeeze(encoder_potentials[0], -1), np.squeeze(encoder_potentials[1], -1), encoder_potentials[2]

    N, T, C = encoder_potentials[0].shape

    def make_fpfun((global_stats, encoder_potentials)):
        return lambda x: \
            _local_ep_update(global_stats, encoder_potentials, x)

    def diff(a, b):
        return np.sum(np.abs(a[0]- b[0]))

    def init_x0():
        fwd_messages = univariate_gaussian.pack_dense(
            np.concatenate([-.5*global_stats[1]*np.ones((N,1)), -.01*np.ones((N, T-1))], axis=-1),
            np.concatenate([np.zeros((N,1)), .01*npr.randn(N, T-1)], axis=-1))
        univariate_gaussian.logZ(fwd_messages[:,0])
        bwd_messages = univariate_gaussian.pack_dense(
            np.concatenate([-.01*np.ones((N, T-1)), np.zeros((N,1))], axis=-1),
            np.concatenate([.01*npr.randn(N, T-1), np.zeros((N,1))], axis=-1))
        obs_messages = univariate_gaussian.pack_dense(-.01*np.ones((N, T)), .01*npr.randn(N,T))
        return fwd_messages, bwd_messages, obs_messages

    return fixed_point(make_fpfun, (global_stats, encoder_potentials), init_x0(), diff, tol=1e-3)

def local_inference(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):

    local_messages = local_inference_messages(global_stats, encoder_potentials)
    local_stats = _local_stats(unbox(global_stats), local_messages)

    local_samples = _dense_local_samples(unbox(global_stats), local_messages, n_samples)
    global_potentials = _global_potentials(local_stats)

    local_kl = _local_kl(unbox(global_stats), local_messages, local_stats)
    global_kl = _global_kl(global_prior_natparams, global_natparams)
    assert(global_kl > 0)

    return local_samples, unbox(global_potentials), global_kl, local_kl

def _bernoulli_loglike(y, logits):
    res = np.sum(np.expand_dims(y, -2)*logits + np.log1p(-sigmoid(logits)))/logits.shape[-2]
    return res

def make_loglike(decoder):
    def loglike(gamma, local_samples, y):
        return _bernoulli_loglike(y, decoder(gamma, local_samples))
    return loglike

def init_pgm_natparams(mean, variance):
    beta = mean/variance
    alpha = mean*beta
    return np.array([alpha-1, -beta])

def pgm_expectedstats(natparams):
    return gamma.expectedstats(natparams)