import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.misc.fixed_points import fixed_point

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

    z_natparams, x_natparams = local_natparams
    z_stats, x_stats = local_stats

    encoder_potentials = gaussian.pack_dense(*encoder_potentials)

    def update_discrete(i):
        h = (i - 1) % K
        j = (i + 1) % K
        ExxT, Ex = gaussian.unpack_dense(x_stats)[0:2]
        ExJxT = np.sum(J_prior[:,:,None,None,...]*ExxT[None,None,...], axis=(-1,-2))
        EmuJmuT = -2*niw_stats[2] # np.sum(mu_prior * mvp(J_prior, mu_prior), axis=-1)
        ExJmuT = np.sum(Ex * h_prior[:,:,None,None,:], axis=-1)
        ElogdetJ = 2*niw_stats[3]

        logodds = z_prior[i] \
            + .5 * (1-z_stats[:,j]) * (ElogdetJ[1,0] - ExJxT[1,0,:,i] - EmuJmuT[1,0] + 2*ExJmuT[1,0,:,i]) \
            + .5 * z_stats[:,j]     * (ElogdetJ[1,1] - ExJxT[1,1,:,i] - EmuJmuT[1,1] + 2*ExJmuT[1,1,:,i]) \
            - .5 * (1-z_stats[:,j]) * (ElogdetJ[0,0] - ExJxT[0,0,:,i] - EmuJmuT[0,0] + 2*ExJmuT[0,0,:,i]) \
            - .5 * z_stats[:,j]     * (ElogdetJ[0,1] - ExJxT[0,1,:,i] - EmuJmuT[0,1] + 2*ExJmuT[0,1,:,i]) \
            + .5 * (1-z_stats[:,h]) * (ElogdetJ[0,1] - ExJxT[0,1,:,h] - EmuJmuT[0,1] + 2*ExJmuT[0,1,:,h]) \
            + .5 * z_stats[:,h]     * (ElogdetJ[1,1] - ExJxT[1,1,:,h] - EmuJmuT[1,1] + 2*ExJmuT[1,1,:,h]) \
            - .5 * (1-z_stats[:,h]) * (ElogdetJ[0,0] - ExJxT[0,0,:,h] - EmuJmuT[0,0] + 2*ExJmuT[0,0,:,h]) \
            - .5 * z_stats[:,h]     * (ElogdetJ[1,0] - ExJxT[1,0,:,h] - EmuJmuT[1,0] + 2*ExJmuT[1,0,:,h])
        return logodds, bernoulli.expectedstats(logodds)

    def update_gaussian(i):
        z_pairstats = _z_pairstats(z_stats)
        natparams_new = encoder_potentials[:,i] + np.tensordot(z_pairstats[:,i], niw_stats_dense, [(1,2), (0,1)])
        return natparams_new, gaussian.expectedstats(natparams_new)

    for i in range(K - 1, -1, -1):
        z_natparams_i, z_stats_i = update_discrete(i)
        z_natparams = replace(z_natparams, z_natparams_i, i, axis=1)
        z_stats = replace(z_stats, z_stats_i, i)
        x_natparams_i, x_stats_i = update_gaussian(i)
        x_natparams = replace(x_natparams, x_natparams_i, i, axis=1)
        x_stats = replace(x_stats, x_stats_i, i, 1)

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

    def log_r(z_i, z_j):
        g_j_left = np.roll(g_j, 1, axis=1)
        g_i_right = np.roll(g_i, -1, axis=1)
        f_i = z_i*z_prior
        f_j = z_j*np.roll(z_prior, -1, axis=0)
        J_prior_i = J_prior[z_i,z_j]
        mu_prior_i = mu_prior[z_i,z_j]
        Psi_i = Psi[:,:,z_i,z_j]
        h_i = np.matmul(J_prior_i, mu_prior_i) + h_rec
        logZ = .5*np.log(np.linalg.det(Psi_i)) + \
            .5*np.log(np.linalg.det(J_prior_i)) + \
            .5*np.sum(mvp(Psi_i, h_i)*h_i, axis=-1) - \
            .5*np.sum(np.matmul(J_prior_i, mu_prior_i)*mu_prior_i)
        return f_i + f_j + z_i*g_j_left + z_j*g_i_right + logZ

    def gen_x_stats():
        log_r_i = np.stack([log_r(0, 0), log_r(0, 1), log_r(1, 0), log_r(1, 1)], axis=-1)
        log_r_i -= logsumexp(log_r_i)[...,None]
        z_pairstats = np.exp(log_r_i).reshape(T,K,2,2)

        m = mvp(Psi, h_prior + h_rec[:,:,None,None,:]) # (T, 2, 2, N)
        stats = gaussian.pack_dense(Psi + outer(m, m), m, np.ones((T,K,2,2)), np.ones((T,K,2,2)))
        return np.sum(z_pairstats[...,None,None] * stats, axis=(2,3))

    log_r_i = np.stack([log_r(0, 0), log_r(0, 1), log_r(1, 0), log_r(1, 1)], axis=-1)
    log_r_i -= logsumexp(log_r_i)[...,None]
    g_j_left = np.roll(g_j, 1, axis=1)
    g_i_right = np.roll(g_i, -1, axis=1)
    g_i = logsumexp(np.stack([log_r_i[..., 2], log_r_i[..., 3]], axis=-1)) - \
            logsumexp(np.stack([log_r_i[..., 0], log_r_i[..., 1]], axis=-1)) - \
            g_j_left - z_prior
    g_j = logsumexp(np.stack([log_r_i[..., 1], log_r_i[..., 3]], axis=-1)) - \
            logsumexp(np.stack([log_r_i[..., 0], log_r_i[..., 2]], axis=-1)) - \
            g_i_right - np.roll(z_prior, -1, axis=0)

    z_natparams = g_i + np.roll(g_j, 1, axis=1) + z_prior
    z_stats = bernoulli.expectedstats(z_natparams)

    x_stats = gen_x_stats()
    x_natparams = gaussian.mean_to_natural(x_stats)

    return (z_natparams, x_natparams), (z_stats, x_stats), (g_i, g_j)

def _local_kl(global_stats, local_natparams, local_stats):
    z_natparams, x_natparams = local_natparams
    z_stats, x_stats = local_stats

    z_kl = _z_kl(global_stats[0], z_natparams, z_stats)
    x_kl = _x_kl(global_stats[1], x_natparams, x_stats, z_stats)

    return z_kl + x_kl

def _global_logZ(global_natparams):
    beta_natparams, niw_natparams = global_natparams
    return beta.logZ(*beta_natparams) + niw.logZ(niw_natparams)

def _global_kl(global_prior_natparams, global_natparams):
    stats = flat((beta.expectedstats(*global_natparams[0]), niw.expectedstats(global_natparams[1])))
    natparam_difference = flat(global_natparams) - flat(global_prior_natparams)
    logZ_difference = _global_logZ(global_natparams) - _global_logZ(global_prior_natparams)
    return np.dot(natparam_difference, stats) - np.sum(logZ_difference)

def _local_inference_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    T, K, N = encoder_potentials[0].shape

    def make_fpfun((global_stats, encoder_potentials)):
        return lambda x: \
            _local_mf_update(global_stats, encoder_potentials, *x)

    def diff(x, x_prev):
        return np.sum(np.abs(x_prev[0][0] - x[0][0])) \
            + np.sum(np.abs(x_prev[0][1][0] - x[0][1][0])) \
            + np.sum(np.abs(x_prev[0][1][1] - x[0][1][1]))

    def init_x0():
        z_natparams = npr.randn(T,K)
        x_natparams = gaussian.pack_dense(-np.tile(np.eye(N), (T, K, 1, 1)), npr.randn(T,K,N))
        z_stats = bernoulli.expectedstats(z_natparams)
        x_stats = gaussian.expectedstats(x_natparams)
        return (z_natparams, x_natparams), (z_stats, x_stats)

    local_natparams, local_stats = fixed_point(make_fpfun, (global_stats, encoder_potentials), init_x0(), diff, tol=1e-3)

    local_samples = gaussian.natural_sample(local_natparams[1], n_samples)
    z_pairstats = _z_pairstats(local_stats[0])

    niw_stats = np.tensordot(z_pairstats, local_stats[1], [(0,1), (0,1)])
    beta_stats = np.sum(local_stats[0], axis=0), np.sum(1 - local_stats[0], axis=0)

    local_kl = _local_kl(unbox(global_stats), local_natparams, local_stats)
    global_kl = _global_kl(global_prior_natparams, global_natparams)
    return local_samples, local_natparams, unbox((beta_stats, niw_stats)), global_kl, local_kl

def local_inference_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    local_samples, _, local_stats, global_kl, local_kl = \
        _local_inference_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples)
    return local_samples, local_stats, global_kl, local_kl

def local_natparams_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials):
    _, local_natparams, _, _, _ = \
        _local_inference_mf(global_prior_natparams, global_natparams, global_stats, encoder_potentials, 0)
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
        local_stats = bernoulli.expectedstats(z_natparams), gaussian.expectedstats(x_natparams)
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

def local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples):
    local_samples, _, local_stats, global_kl, local_kl = \
        _local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, n_samples)
    return local_samples, local_stats, global_kl, local_kl

def local_natparams_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials):
    _, local_natparams, _, _, _ = \
        _local_inference_ep(global_prior_natparams, global_natparams, global_stats, encoder_potentials, 0)
    return local_natparams

def init_pgm_natparams(K, N, alpha, niw_conc=10., random_scale=0.):
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
    return beta.expectedstats(*natparams[0]), niw.expectedstats(natparams[1])
