from svae.distributions import univariate_gaussian
from svae.models.univariate_lds import init_pgm_natparams, local_inference, local_inference_messages, pgm_expectedstats, _dense_local_samples
from svae.nnet import gaussian_mean, init_layer_glorot, init_gresnet, init_gresnet_mix, make_loglike
from svae.optimizers import adam
from svae.svae import make_gradfun
from svae.util import softmax
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats

def likelihood(loc, scale, x):
    # return .25*(np.sin(x*scale))+.5
    return scipy.stats.laplace.pdf(x, loc=loc, scale=scale)

def gaussian_likelihood(x, mean, sigmasq):
    mean, sigmasq = np.squeeze(mean,-1), np.squeeze(sigmasq,-1)
    return (1./np.sqrt(2*np.pi*sigmasq))*np.exp(-.5*(x-mean)**2/sigmasq)

def gen_data(N, T, tau, scale):
    x = np.cumsum(npr.randn(N,T)*np.sqrt(1./tau), axis=1)[...,None]
    # y = x + scipy.stats.laplace.rvs(scale=scale)
    y = x + scipy.stats.laplace.rvs(scale=scale, size=(N,T,1))
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
    cavity_v, cavity_m = univariate_gaussian.natural_to_standard((fwd_messages + bwd_messages)[0,t])
    cavity_dist = scipy.stats.norm(loc=cavity_m, scale=np.sqrt(cavity_v))

    post_v, post_m = univariate_gaussian.natural_to_standard((fwd_messages + bwd_messages + obs_messages)[0,t])
    post_dist = scipy.stats.norm(loc=post_m, scale=np.sqrt(post_v))

    xs = np.linspace(-5, 5, 100)

    normalize = lambda _: _/np.max(_)

    plot_or_update(0, ax, xs, normalize(post_dist.pdf(xs)), linestyle='-', linewidth=2, color='m')
    # plot_or_update(1, ax, xs, normalize(cavity_dist.pdf(xs)), linestyle='-', linewidth=2, color='k')

    plot_or_update(1, ax, xs, normalize(gaussian_likelihood(y[n,t], *decoder(gamma, xs[..., None]))), linestyle='-', linewidth=2, color='r')
    plot_or_update(2, ax, xs, normalize(likelihood(y[n,t], scale, xs)), linestyle='-', linewidth=2, color='g')
    plot_or_update(3, ax, [x[n, t], x[n, t]], [0, 1], linestyle='-', linewidth=1, color='k')
    plot_or_update(4, ax, [y[n, t], y[n, t]], [0, 1], linestyle='-', linewidth=1, color='r')

    # encoder_pdf = np.zeros(xs.shape)

    for i in xrange(x_encW.shape[-1]):
        encoder_dist = scipy.stats.norm(loc=x_enc[0,t,i], scale=np.sqrt(x_encV[0,t,i]))
        # encoder_pdf = encoder_pdf + encoder_dist.pdf(xs)*x_encW[0,t,i]
        plot_or_update(5+i, ax, xs, normalize(encoder_dist.pdf(xs)), linestyle='-', linewidth=2, color='b', alpha=x_encW[0,t,i])
    # plot_or_update(3, ax, xs, normalize(encoder_pdf), linestyle='-', linewidth=2, color='b')

def plot_time_series(global_natparams, encoder, phi, decoder, gamma, scale, x, y, n, ax):
    T = y.shape[1]
    plot_or_update(0, ax, np.arange(0,T), x[n], linestyle='-', linewidth=2, color='r')
    plot_or_update(1, ax, np.arange(0,T), y[n], marker='.', linestyle='', color='k')

    j = 1./np.power(scale, 2)
    true_likelihood = -.5*j*np.ones((1,T,1,1)), j*y[n].reshape((1,T,1,1)), np.zeros((1,T,1))

    encoder_out = encoder(phi, y[n:n+1])

    global_stats = pgm_expectedstats(global_natparams)

    fwd_messages, bwd_messages, obs_messages = local_inference_messages(global_stats, encoder_out)
    posterior_m = univariate_gaussian.natural_to_standard(fwd_messages + bwd_messages + obs_messages)[1]
    plot_or_update(2, ax, np.arange(0, T), posterior_m.reshape(T), linestyle='-', linewidth=2, color='b')

    local_samples = _dense_local_samples(global_stats, (fwd_messages, bwd_messages, obs_messages), 1)
    decoder_out = decoder(gamma, local_samples)
    decoder_samples = univariate_gaussian.natural_sample(
        univariate_gaussian.standard_to_natural(decoder_out[1], decoder_out[0]), 1)
    plot_or_update(3, ax, np.arange(0, T), decoder_samples.reshape(T), marker='.', linestyle='', color='r')

    fwd_messages, bwd_messages, obs_messages = local_inference_messages(global_stats, true_likelihood)
    posterior_m = univariate_gaussian.natural_to_standard(fwd_messages + bwd_messages + obs_messages)[1]
    plot_or_update(4, ax, np.arange(0, T), posterior_m.reshape(T), linestyle='-', linewidth=2, color='g')

def make_plotter(encoder, decoder, global_prior_natparams, scale, x, y):
    N, T, _ = y.shape
    figure, axes = plt.subplots(3, 2, figsize=(8,3))

    # latent space
    for i in range(3):
        axes[i,0].set_ylim(0,1)
        axes[i,0].set_xlim(-5,5)
        axes[i,0].autoscale(False)
        axes[i,1].set_ylim(-2,2)
        axes[i,1].set_xlim(0,T)
        axes[i,1].autoscale(True)

    figure.tight_layout()

    def plot(iter, val, params, _):
        global_natparams, gamma, phi = params
        print('{}: {}, tau={}'.format(iter, val, pgm_expectedstats(global_natparams)[1]))
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 0, 25, axes[0,0])
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 1, 25, axes[1,0])
        plot_encoder_potential(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 2, 25, axes[2,0])
        plot_time_series(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 0, axes[0,1])
        plot_time_series(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 1, axes[1,1])
        plot_time_series(global_natparams, encoder, phi, decoder, gamma, scale, x, y, 2, axes[2,1])
        plt.pause(0.1)

    return plot

if __name__ == "__main__":
    # use constant data set
    npr.seed(0)

    N = 10000
    T = 50
    C = 2
    tau = 20.0
    scale = .5
    x, y = gen_data(N, T, tau, scale)

    # global prior and variational posterior natparams
    global_prior_natparams = init_pgm_natparams(tau, np.power(20.0, 2))
    global_natparams = init_pgm_natparams(tau, np.power(50.0, 2))

    # construct recognition and decoder networks and initialize them
    encoder, phi = \
        init_gresnet_mix(1, 1, C, [
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot)])
    decoder, gamma = \
        init_gresnet(1, [
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot),
            (20, np.tanh, init_layer_glorot),
            (2*1, gaussian_mean)])

    loglike = make_loglike(decoder)

    params = global_natparams, gamma, phi

    plotter = make_plotter(encoder, decoder, global_prior_natparams, scale, x, y)

    np.seterr(all='raise')

    gradfun = make_gradfun(
        local_inference, encoder, loglike, global_prior_natparams, pgm_expectedstats, y)

    params = adam(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e4, callback=plotter),
                  params, num_iters=5000, step_size=1e-3)