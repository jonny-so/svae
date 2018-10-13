from svae.distributions import univariate_gaussian
from svae.models.dlds import \
    init_pgm_natparams, local_inference, local_inference_messages, make_loglike, pgm_expectedstats, _dense_local_samples
from svae.nnet import init_layer_glorot, init_mlp, init_gresnet_mix
from svae.optimizers import adam
from svae.svae import make_gradfun
from svae.util import sigmoid, softmax
from toolz import identity
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats
import pickle

def likelihood(scale, x):
    # return .25*(np.sin(x*scale))+.5
    return scipy.stats.norm.cdf(x, scale=scale)

def gen_data(N, T, tau, scale):
    x = np.cumsum(npr.randn(N,T)*np.sqrt(1./tau), axis=1)[...,None]
    y = np.array(npr.rand(*x.shape) < likelihood(scale, x), dtype=np.int32)
    return x, y

def plot_or_update(idx, ax, x, y, alpha=1, **kwargs):
    if len(ax.lines) > idx:
        ax.lines[idx].set_data((x, y))
        ax.lines[idx].set_alpha(alpha)
    else:
        ax.plot(x, y, alpha=alpha, **kwargs)

def plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, n, t, ax):
    encoder_out = encoder(phi, y[n:n+1])

    x_encV, x_enc = univariate_gaussian.natural_to_standard(
        univariate_gaussian.pack_dense(np.squeeze(encoder_out[0], -1), np.squeeze(encoder_out[1], -1)))
    x_encW = softmax(encoder_out[2])

    fwd_messages, bwd_messages, obs_messages = local_inference_messages(pgm_expectedstats(global_natparams), encoder_out)
    post_v, post_m = univariate_gaussian.natural_to_standard((fwd_messages + bwd_messages + obs_messages)[0,t])
    post_dist = scipy.stats.norm(loc=post_m, scale=np.sqrt(post_v))

    xs = np.linspace(-2., 2)

    normalize = lambda _: _/np.max(_)

    plot_or_update(0, ax, xs, normalize(post_dist.pdf(xs)), linestyle='-', linewidth=2, color='m')

    if y[n,t] == 1:
        plot_or_update(1, ax, xs, sigmoid(decoder(gamma, xs[..., None])), linestyle='-', linewidth=2, color='r')
        plot_or_update(2, ax, xs, likelihood(scale, xs), linestyle='-', linewidth=2, color='g')
    else:
        plot_or_update(1, ax, xs, 1-sigmoid(decoder(gamma, xs[..., None])), linestyle='-', linewidth=2, color='r')
        plot_or_update(2, ax, xs, 1-likelihood(scale, xs), linestyle='-', linewidth=2, color='g')
    encoder_pdf = np.zeros(xs.shape)

    for i in xrange(x_encW.shape[-1]):
        encoder_dist = scipy.stats.norm(loc=x_enc[0,t,i], scale=np.sqrt(x_encV[0,t,i]))
        encoder_pdf = encoder_pdf + encoder_dist.pdf(xs)*x_encW[0,t,i]

    plot_or_update(3, ax, xs, normalize(encoder_pdf), linestyle='-', linewidth=2, color='b')
    plot_or_update(4, ax, [x[n,t], x[n,t]], [0,1], linestyle='-', linewidth=1, color='k')

def plot_time_series(global_natparams, encoder, phi, x, y, n, ax):
    T = y.shape[1]
    plot_or_update(0, ax, np.arange(0,T), x[n], linestyle='-', linewidth=2, color='k')

    encoder_out = encoder(phi, y[n:n+1])

    global_stats = pgm_expectedstats(global_natparams)

    fwd_messages, bwd_messages, obs_messages = local_inference_messages(global_stats, encoder_out)
    posterior_v, posterior_m = univariate_gaussian.natural_to_standard(fwd_messages + bwd_messages + obs_messages)
    plot_or_update(1, ax, np.arange(0, T), posterior_m.reshape(T), linestyle='-', linewidth=2, color='g')

    if len(ax.collections) > 0:
        ax.collections.pop()
    lb = np.reshape(posterior_m - np.sqrt(posterior_v), T)
    ub = np.reshape(posterior_m + np.sqrt(posterior_v), T)
    ax.fill_between(np.arange(0, T), lb, ub, facecolor='orange', alpha=0.75)

def validation_error(global_natparams, encoder, phi, decoder, gamma, y, n_samples = 100):
    encoder_out = encoder(phi, y[:,:-1])

    global_stats = pgm_expectedstats(global_natparams)

    fwd_messages, bwd_messages, obs_messages = local_inference_messages(global_stats, encoder_out)
    posterior_v, posterior_m = univariate_gaussian.natural_to_standard(fwd_messages + bwd_messages + obs_messages)

    x_pred = univariate_gaussian.natural_sample(
        univariate_gaussian.standard_to_natural(posterior_v[:,-1] + 1./global_stats[1], posterior_m[:,-1]), n_samples)

    p_pred = np.mean(sigmoid(decoder(gamma, x_pred[:,:,None])), axis=0)

    return -np.mean(y[:,-1]*np.log(p_pred) + (1-y[:,-1])*np.log(1-p_pred))

def save_likelihood_plot(fileprefix, scale, y, ):

    fig, ax = plt.subplots(1, 1, figsize=(4*4, 4))
    ax.set_xlim(-3,3)
    ax.set_ylim(0,1.1)
    ax.set_aspect('equal')
    ax.autoscale(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    xs = np.linspace(-4., 4, 200)
    if y == 1:
        plot_or_update(0, ax, xs, likelihood(scale, xs), linestyle='-', linewidth=2, color='g')
    else:
        plot_or_update(0, ax, xs, 1 - likelihood(scale, xs), linestyle='-', linewidth=2, color='g')

    filename = fileprefix + '_likelihood_' + 'y'+ str(y) + '_.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print 'saved {}'.format(filename)
    plt.close('all')

def save_encoder_decoder_plot(fileprefix, encoder, phi, decoder, gamma, y):

    fig, ax = plt.subplots(1, 1, figsize=(4*4, 4))
    ax.set_xlim(-3,3)
    ax.set_ylim(0,1.1)
    ax.set_aspect('equal')
    ax.autoscale(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    encoder_out = encoder(phi, y*np.ones((1,1,1)))

    x_encV, x_enc = univariate_gaussian.natural_to_standard(
        univariate_gaussian.pack_dense(np.squeeze(encoder_out[0], -1), np.squeeze(encoder_out[1], -1)))
    x_encW = softmax(encoder_out[2])

    xs = np.linspace(-4., 4, 200)
    normalize = lambda _: _/np.max(_)

    if y == 1:
        plot_or_update(0, ax, xs, sigmoid(decoder(gamma, xs[..., None])), linestyle='-', linewidth=2, color='r')
    else:
        plot_or_update(0, ax, xs, 1-sigmoid(decoder(gamma, xs[..., None])), linestyle='-', linewidth=2, color='r')

    encoder_pdf = np.zeros(xs.shape)

    for i in xrange(x_encW.shape[-1]):
        encoder_dist = scipy.stats.norm(loc=x_enc[0,0,i], scale=np.sqrt(x_encV[0,0,i]))
        encoder_pdf = encoder_pdf + encoder_dist.pdf(xs)*x_encW[0,0,i]

    ax.fill_between(xs, np.zeros(xs.shape), normalize(encoder_pdf), facecolor='blue', alpha=0.5)

    filename = fileprefix + '_encoder_decoder_' + 'y'+ str(y) + '_.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print 'saved {}'.format(filename)
    plt.close('all')

def make_plotter(encoder, decoder, global_prior_natparams, scale, x, y):
    T, K, _ = y.shape
    figure, axes = plt.subplots(3, 2, figsize=(8,3))

    # latent space
    for i in range(3):
        axes[i,0].set_ylim(0,1)
        axes[i,0].set_xlim(-2,2)
        axes[i,0].set_aspect('equal')
        axes[i,0].autoscale(False)
        axes[i,1].set_ylim(-2,2)
        axes[i,1].set_xlim(0,T)
        axes[i,1].autoscale(True)

    figure.tight_layout()

    def plot(iter, val, params, _):
        global_natparams, gamma, phi = params
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 0, 50, axes[0,0])
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 1, 50, axes[1,0])
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 2, 50, axes[2,0])
        plot_time_series(global_natparams, encoder, phi, x, y, 0, axes[0,1])
        plot_time_series(global_natparams, encoder, phi, x, y, 1, axes[1,1])
        plot_time_series(global_natparams, encoder, phi, x, y, 2, axes[2,1])
        plt.pause(0.1)

    return plot

def run_experiment(C, seed):
    # use constant data set
    npr.seed(seed)

    n_iter = 2000
    N = 10000
    T = 100
    tau = 100.0
    scale = .5
    x, y = gen_data(N, T, tau, scale)
    x_valid, y_valid = gen_data(1000, T+1, tau, scale)

    # global prior and variational posterior natparams
    global_prior_natparams = init_pgm_natparams(tau, np.power(20.0, 2))
    global_natparams = init_pgm_natparams(tau, np.power(50.0, 2))

    # construct recognition and decoder networks and initialize them
    encoder, phi = \
        init_gresnet_mix(1, 1, C, [
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot)])
    decoder, gamma = \
        init_mlp(1, [
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot),
            (1, identity, init_layer_glorot)])
    loglike = make_loglike(decoder)

    params = global_natparams, gamma, phi

    plot = make_plotter(encoder, decoder, global_prior_natparams, scale, x, y)
    elbos = np.zeros(n_iter//10)
    errs = np.zeros(n_iter//10)

    def callback(i, val, params, grad):
        if i % 10 == 0:
            plot(i, val, params, grad)
            err = validation_error(params[0], encoder, phi, decoder, gamma, y_valid, 100)
            print('{}: elbo={}, err={}'.format(i, val, err))
            elbos[i // 10] = val
            errs[i // 10] = err

    np.seterr(all='raise')

    gradfun = make_gradfun(
        local_inference, encoder, loglike, global_prior_natparams, pgm_expectedstats, y)

    params = adam(gradfun(batch_size=50, num_samples=3, natgrad_scale=1e4, callback=callback),
                  params, num_iters=n_iter, step_size=1e-3)

    fileprefix = 'binary_lds/binary_lds_c' + str(C) + '_s' + str(seed)
    pickle.dump(params, open(fileprefix + '_params.pkl', 'wb'))
    save_encoder_decoder_plot(fileprefix, encoder, phi, decoder, gamma, 0)
    save_encoder_decoder_plot(fileprefix, encoder, phi, decoder, gamma, 1)
    save_likelihood_plot(fileprefix, scale, 0)
    save_likelihood_plot(fileprefix, scale, 1)
    np.savetxt(fileprefix + '_elbo.txt', elbos)
    np.savetxt(fileprefix + '_errs.txt', errs)

if __name__ == "__main__":
    run_experiment(2, 0)
    run_experiment(2, 1)
    run_experiment(2, 2)
    run_experiment(2, 3)
    run_experiment(2, 4)