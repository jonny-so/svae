import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.fixed_points import fixed_point
from autograd.core import getval
from toolz import curry

from svae.distributions import bernoulli, beta, gaussian, niw
from svae.util import flat, expand_diagonal, logsumexp, mvp, replace, outer, symmetrize, unbox

def _z_pairstats(z_stats):
    return np.stack([
        (1 - z_stats) * (1 - np.roll(z_stats, -1, axis=1)),
        (1 - z_stats) * np.roll(z_stats, -1, axis=1),
        z_stats * (1 - np.roll(z_stats, -1, axis=1)),
        z_stats * np.roll(z_stats, -1, axis=1)], axis=-1).reshape(z_stats.shape + (2,2))

def _z_kl(beta_stats, z_natparams, z_stats):
    z_prior_natparams = beta_stats[0] - beta_stats[1]

    kl = (z_natparams - z_prior_natparams) * z_stats \
        - bernoulli.logZ(z_natparams) - beta_stats[1]

    assert (kl.shape == z_natparams.shape)
    assert(np.all(np.isfinite(z_natparams)))
    assert(np.all(np.isfinite(z_prior_natparams)))
    assert(np.allclose(z_stats, bernoulli.natural_to_mean(z_natparams)))
    assert(np.all(z_stats > 0))
    assert(np.all(z_stats <= 1))
    assert(np.all(kl >= 0))
    return np.sum(kl)

def _x_kl(niw_stats, x_natparams, x_stats, z_stats):
    global_potentials = np.tensordot(_z_pairstats(z_stats), niw_stats, [(2,3), (0,1)])
    return np.tensordot(x_natparams - global_potentials, x_stats, 4) - gaussian.logZ(x_natparams)

def _local_mf_update(global_stats, encoder_potentials, local_natparams, local_stats):

    T,K,N = encoder_potentials[1].shape

    beta_stats, niw_stats = global_stats[0], gaussian.unpack_dense(global_stats[1])
    niw_stats_dense = global_stats[1]
    z_prior = beta_stats[0] - beta_stats[1]
    J_prior, h_prior = -2*symmetrize(niw_stats[0]), niw_stats[1]

    # mu_prior, _ = gaussian.natural_to_mean(h_prior, -.5*J_prior)
    z_natparams, x_natparams = local_natparams
    z_stats, x_stats = local_stats

    J_rec, h_rec = -2*expand_diagonal(encoder_potentials[0]), encoder_potentials[1]

    def surrogate_elbo(local_natparams, local_stats):
        local_kl = _local_kl(global_stats, local_natparams, local_stats)
        other = np.tensordot(local_stats[1], encoder_potentials, 4)
        return local_kl + other

    def update_discrete(i):
        h = (i - 1) % K
        j = (i + 1) % K

        ExJxT = np.sum(J_prior[:,:,None,None,...]*x_stats[1][None,None,...], axis=(-1,-2))
        muJmuT = -2*niw_stats[2] # np.sum(mu_prior * mvp(J_prior, mu_prior), axis=-1)
        ExJmuT = np.sum(x_stats[0] * h_prior[:,:,None,None,:], axis=-1)
        ElogdetJ = 2*niw_stats[3]
        assert(ExJxT.shape == (2,2,T,K))
        assert(muJmuT.shape == (2,2))
        assert(ExJmuT.shape == (2,2,T,K))

        logodds = z_prior[i] \
            + .5 * (1-z_stats[:,j]) * (ElogdetJ[1,0] - ExJxT[1,0,:,i] - muJmuT[1,0] + 2*ExJmuT[1,0,:,i]) \
            + .5 * z_stats[:,j]     * (ElogdetJ[1,1] - ExJxT[1,1,:,i] - muJmuT[1,1] + 2*ExJmuT[1,1,:,i]) \
            - .5 * (1-z_stats[:,j]) * (ElogdetJ[0,0] - ExJxT[0,0,:,i] - muJmuT[0,0] + 2*ExJmuT[0,0,:,i]) \
            - .5 * z_stats[:,j]     * (ElogdetJ[0,1] - ExJxT[0,1,:,i] - muJmuT[0,1] + 2*ExJmuT[0,1,:,i]) \
            + .5 * (1-z_stats[:,h]) * (ElogdetJ[0,1] - ExJxT[0,1,:,h] - muJmuT[0,1] + 2*ExJmuT[0,1,:,h]) \
            + .5 * z_stats[:,h]     * (ElogdetJ[1,1] - ExJxT[1,1,:,h] - muJmuT[1,1] + 2*ExJmuT[1,1,:,h]) \
            - .5 * (1-z_stats[:,h]) * (ElogdetJ[0,0] - ExJxT[0,0,:,h] - muJmuT[0,0] + 2*ExJmuT[0,0,:,h]) \
            - .5 * z_stats[:,h]     * (ElogdetJ[1,0] - ExJxT[1,0,:,h] - muJmuT[1,0] + 2*ExJmuT[1,0,:,h])
        return logodds, bernoulli.natural_to_mean(logodds)

    def update_gaussian(i):
        z_pairstats = _z_pairstats(z_stats)
        natparams_new = x_natparams[:,i] + np.tensordot(z_pairstats[:,i], niw_stats_dense, [(1,2), (0,1)])
        return natparams_new, gaussian.expectedstats(natparams_new)

    for i in range(K - 1, -1, -1):
        z_natparams_i, z_stats_i = update_discrete(i)
        z_natparams = replace(z_natparams, z_natparams_i, i)
        z_stats = replace(z_stats, z_stats_i, i)
        x_natparams_i, x_stats_i = update_gaussian(i)
        x_natparams = replace(x_natparams, x_natparams_i, i, 1)
        x_stats = replace(x_stats, x_stats_i, i, 1)

    # zkl = _z_kl(z_natparams, z_prior, z_stats)
    # xkl = _x_kl(x_natparams, J_prior, mu_prior, x_stats, z_stats)
    # print('x: {}, z: {}, sum: {}'.format(getval(xkl), getval(zkl), getval(xkl+zkl)))
    # print('z_stats: \n{}'.format(getval(z_stats[0])))
    # print('x_stats: \n{}'.format(getval(x_stats[0][0])))
    # print('x_stats: \n{}'.format(getval(x_stats[1][0])))
    # print('z_nats: \n{}'.format(getval(z_natparams[0])))
    # print('x_nats[0: \n{}'.format(getval(x_natparams[0][0])))
    # print('x_nats[1]: \n{}'.format(getval(x_natparams[1][0])))
    # assert(zkl > 0)
    # assert(xkl > 0)

    return (z_natparams, x_natparams), (z_stats, x_stats)

def _local_ep_update(global_stats, encoder_potentials, local_messages):

    T,K,N = encoder_potentials[1].shape

    beta_stats, niw_stats = global_stats[0], gaussian.unpack_dense(global_stats[1])

    z_prior = beta_stats[0] - beta_stats[1]
    J_prior, h_prior = -2*symmetrize(niw_stats[0]), niw_stats[1]
    mu_prior = np.linalg.solve(J_prior, h_prior)

    J_rec, h_rec = -2*expand_diagonal(encoder_potentials[0]), encoder_potentials[1]
    mu_rec = np.linalg.solve(J_rec, h_rec)
    Psi = np.linalg.inv(J_rec[...,None,None,:,:] + J_prior)

    g_i, g_j = local_messages

    def log_r(i, z_i, z_j, g_j_left, g_i_right):
        j = (i + 1) % K
        f_i = z_i*z_prior[i]
        f_j = z_j*z_prior[j]
        J_prior_i = J_prior[z_i,z_j]
        mu_prior_i = mu_prior[z_i,z_j]
        Psi_i = Psi[:,i,z_i,z_j]
        h_i = np.matmul(J_prior_i, mu_prior_i) + h_rec[:,i,:]
        logZ = .5*np.log(np.linalg.det(Psi[:,i,z_i,z_j])) + \
            .5*np.log(np.linalg.det(J_prior_i)) + \
            .5*np.sum(mvp(Psi_i, h_i)*h_i, axis=1) - \
            .5*np.sum(np.matmul(J_prior_i, mu_prior_i)*mu_prior_i)
        return f_i + f_j + z_i*g_j_left + z_j*g_i_right + logZ

    def update(i, g_j_left, g_i_right):
        j = (i + 1) % K
        h = (i - 1) % K
        log_r_i = np.stack([
            log_r(i, 0, 0, g_j_left, g_i_right),
            log_r(i, 0, 1, g_j_left, g_i_right),
            log_r(i, 1, 0, g_j_left, g_i_right),
            log_r(i, 1, 1, g_j_left, g_i_right)]).T
        log_r_i -= logsumexp(log_r_i)[...,None]
        g_i_i = logsumexp(np.stack([log_r_i[:, 2], log_r_i[:, 3]]).T) - \
                logsumexp(np.stack([log_r_i[:, 0], log_r_i[:, 1]]).T) - \
                g_j_left - z_prior[i]
        g_j_i = logsumexp(np.stack([log_r_i[:, 1], log_r_i[:, 3]]).T) - \
                logsumexp(np.stack([log_r_i[:, 0], log_r_i[:, 2]]).T) - \
                g_i_right - z_prior[j]
        return g_i_i, g_j_i

    def gen_x_stats(i):
        h = (i - 1) % K
        j = (i + 1) % K
        log_r_i = np.stack([
            log_r(i, 0, 0, g_j[:, h], g_i[:, j]),
            log_r(i, 0, 1, g_j[:, h], g_i[:, j]),
            log_r(i, 1, 0, g_j[:, h], g_i[:, j]),
            log_r(i, 1, 1, g_j[:, h], g_i[:, j])]).T
        log_r_i -= logsumexp(log_r_i)[...,None]
        z_pairstats = np.exp(log_r_i).reshape(T,2,2)

        m = mvp(Psi[:,i], h_prior + h_rec[:,i,None,None,:]) # (T, 2, 2, N)
        stats = gaussian.pack_dense(Psi[:,i] + outer(m, m), m, np.ones((T,2,2)), np.ones((T,2,2)))
        return np.sum(z_pairstats[...,None,None] * stats, axis=(1,2))

    for i in range(K):
        h = (i - 1) % K
        j = (i + 1) % K
        g_i_i, g_j_i = update(i, g_j[:,h], g_i[:,j])
        g_i = replace(g_i, g_i_i, i)
        g_j = replace(g_j, g_j_i, i)

    z_natparams = g_i + np.roll(g_j, 1, axis=1) + z_prior
    z_stats = bernoulli.natural_to_mean(z_natparams)

    x_stats = np.stack([gen_x_stats(i) for i in range(K)], axis=1)
    x_natparams = gaussian.mean_to_natural(x_stats)

    # assert(np.all(getval(z_stats) > 0))
    # assert(np.all(getval(z_stats) < 1))


    # zkl = _z_kl(z_natparams, z_prior, z_stats)
    # xkl = _x_kl(x_natparams, J_prior, mu_prior, x_stats, z_stats)
    # assert(zkl > 0)
    # assert(xkl > 0)
    # print('x: {}, z: {}, sum: {}'.format(getval(xkl), getval(zkl), getval(xkl+zkl)))
    # print('z_stats: \n{}'.format(getval(z_stats[0])))
    # print('x_stats: \n{}'.format(getval(x_stats[0][0])))
    # print('x_stats: \n{}'.format(getval(x_stats[1][0])))
    # print('z_nats: \n{}'.format(getval(z_natparams[0])))
    # print('x_nats[0: \n{}'.format(getval(x_natparams[0][0])))
    # print('x_nats[1]: \n{}'.format(getval(x_natparams[1][0])))
    # print('z_stats {} {}'.format(np.mean(z_stats), np.mean(np.abs(z_stats))))
    # print('z: {}, x: {}'.format(zkl, xkl))
    return (z_natparams, x_natparams), (z_stats, x_stats), (g_i, g_j) #, zkl + xkl

def _local_kl(global_stats, local_natparams, local_stats):
    z_natparams, x_natparams = local_natparams
    z_stats, x_stats = local_stats

    z_kl = _z_kl(global_stats[0], z_natparams, z_stats)
    x_kl = _x_kl(global_stats[1], x_natparams, x_stats, z_stats)
    print('z: {}, x: {}'.format(getval(z_kl), getval(x_kl)))
    return z_kl + x_kl

def _global_logZ(global_natparams):
    beta_natparams, niw_natparams = global_natparams
    return beta.logZ(*beta_natparams) + niw.logZ(niw_natparams)

def _global_kl(global_prior_natparams, global_natparams):
    stats = flat((beta.natural_to_mean(*global_natparams[0]), niw.expectedstats(global_natparams[1])))
    natparam_difference = flat(global_natparams) - flat(global_prior_natparams)
    logZ_difference = _global_logZ(global_natparams) - _global_logZ(global_prior_natparams)
    return np.dot(natparam_difference, stats) - np.sum(logZ_difference)

def local_inference_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    T, K, N = encoder_potentials[0].shape

    def make_fpfun((global_stats, encoder_potentials)):
        return lambda x: \
            _local_mf_update(global_stats, encoder_potentials, *x)

    def diff(x, x_prev):
        return np.sum(np.abs(x_prev[0][0] - x[0][0])) \
            + np.sum(np.abs(x_prev[0][1][0] - x[0][1][0])) \
            + np.sum(np.abs(x_prev[0][1][1] - x[0][1][1]))

    def init_x0():
        z_natparams = npr.randn(T, K)
        x_natparams = gaussian.pack_dense(-np.tile(np.eye(N), (T, K, 1, 1)), npr.randn(T, K, N))
        z_stats = bernoulli.natural_to_mean(z_natparams)
        x_stats = gaussian.expectedstats(x_natparams)
        return (z_natparams, x_natparams), (z_stats, x_stats)

    local_natparams, local_stats = fixed_point(make_fpfun, (global_stats, encoder_potentials), init_x0(), diff, tol=1e-3)

    local_samples = gaussian.natural_sample(local_natparams[1], n_samples)
    z_pairstats = _z_pairstats(local_stats[0])

    niw_stats = np.tensordot(z_pairstats, local_stats[1], [(0,1), (0,1)])
    beta_stats = np.sum(local_stats[0], axis=0), np.sum(1 - local_stats[0], axis=0)

    local_kl = _local_kl(unbox(global_stats), local_natparams, local_stats)
    global_kl = _global_kl(global_prior_natparams, global_natparams)
    return local_samples, unbox((beta_stats, niw_stats)), global_kl, local_kl


def local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    local_samples, _, local_stats, global_kl, local_kl = \
        _local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples)
    return local_samples, local_stats, global_kl, local_kl

def local_natparams_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials):
    _, local_natparams, _, _, _ = \
        _local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, 0)
    return local_natparams

def _local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    T, K, N = encoder_potentials[0].shape

    def make_fpfun((global_stats, encoder_potentials)):
        return lambda x: \
            _local_ep_update(global_stats, encoder_potentials, x[2])

    def diff(x, x_prev):
        return np.sum(np.abs(x_prev[0][0] - x[0][0])) \
            + np.sum(np.abs(x_prev[0][1][0] - x[0][1][0])) \
            + np.sum(np.abs(x_prev[0][1][1] - x[0][1][1]))

    def init_x0():
        z_natparams = npr.randn(T, K)
        x_natparams = gaussian.pack_dense(-np.tile(np.eye(N), (T, K, 1, 1)), npr.randn(T, K, N))
        local_stats = bernoulli.natural_to_mean(z_natparams), gaussian.expectedstats(x_natparams)
        local_messages = npr.randn(T,K), npr.randn(T,K)
        return (z_natparams, x_natparams), local_stats, local_messages

    local_natparams, local_stats, _ = fixed_point(make_fpfun, (global_stats, encoder_potentials), init_x0(), diff, tol=1e-3)
    local_samples = gaussian.natural_sample(local_natparams[1], n_samples)
    z_pairstats = _z_pairstats(local_stats[0])

    niw_stats = np.tensordot(z_pairstats, local_stats[1], [(0,1), (0,1)])
    beta_stats = np.sum(local_stats[0], axis=0), np.sum(1-local_stats[0], axis=0)

    local_kl = _local_kl(unbox(global_stats), local_natparams, local_stats)
    global_kl = _global_kl(global_prior_natparams, global_natparams)
    return local_samples, local_natparams, unbox((beta_stats, niw_stats)), global_kl, local_kl

def init_global_natparams(K, N, alpha, niw_conc=10., random_scale=0.):
    def init_niw_natparam():
        nu, kappa = np.ones((2,2))*(N+niw_conc), np.ones((2,2))*niw_conc
        # m = np.zeros((2,2,N))
        S = .5*np.tile(np.eye(N), (2,2,1,1))
        #.25*np.tile((N+niw_conc)*np.eye(N), (2,2,1,1))
        m = 1.25*np.array([[[0,1],[1,0]],[[0,-1],[-1,0]]]) # m + random_scale * npr.randn(*m.shape)
        return niw.standard_to_natural(S, m, kappa, nu)

    beta_natparam = alpha * (npr.rand(K,2) if random_scale else np.ones((K,2)))
    niw_natparam = init_niw_natparam()

    return (beta_natparam[:,0], beta_natparam[:,1]), niw_natparam

def pgm_expectedstats(natparams):
    return beta.natural_to_mean(*natparams[0]), niw.expectedstats(natparams[1])
