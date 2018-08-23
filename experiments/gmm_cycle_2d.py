import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.stats
import pickle
from matplotlib.colors import LinearSegmentedColormap

from svae.nnet import init_gresnet, gaussian_mean, gaussian_info, make_loglike
from svae.distributions import beta, gaussian, niw
from svae.models.gmm_cycle import init_global_natparams, local_inference_ep, local_natparams_ep, pgm_expectedstats
from svae.optimizers import adam
from svae.svae import make_gradfun
from svae.util import expand_diagonal, mvp

# alpha, beta - (shared) beta prior hyper-parameters for pi
# K - length of cycle / number of dependent observations
# N - number of independent samples
def create_gmm_cycle_2d_data(alpha, beta, K, T):

    N = P = 2 # N latent dimensions, P observed dimensions
    pi = npr.beta(alpha, beta, K)
    mu = 1.25*np.array([[[0,1],[1,0]],[[0,-1],[-1,0]]])

    V = .0075*np.array([
        [[[1, 0], [0, 15]], [[15, 0], [0, 1]]],
        [[[1, 0], [0, 15]], [[15, 0], [0, 1]]]])
    z = npr.binomial(1, pi, (T, K))

    mu_x = mu[z, np.roll(z, -1, axis=1)]
    for t in range(T):
        for i in range(K):
            j = (i + 1) % K
            assert(np.allclose(mu_x[t,i], mu[z[t,i], z[t,j]]))
    V_x = V[z, np.roll(z, -1, axis=1)]
    J_x = np.linalg.inv(V_x)

    x = np.squeeze(gaussian.natural_sample(gaussian.pack_dense(-.5*J_x, mvp(J_x, mu_x)), 1), -2)

    def warp(u, v, x0=0, y0=0):
        r = np.sqrt((u- x0)**2 + (v - y0)**2)
        t = np.pi*r/4 - .5*np.pi
        x = (u - x0)*np.cos(t) + (v - y0)*np.sin(t) + x0
        y = -(u - x0)*np.sin(t) + (v - y0)*np.cos(t) + y0
        return np.stack((x, y), axis=2)

    y = warp(x[...,0], x[...,1], 0, 0)

    return pi, mu, V, y, z

colors = np.array([
    [106,61,154],  # Dark colors
    [31,120,180],
    [51,160,44],
    [227,26,28],
    [255,127,0],
    [166,206,227],  # Light colors
    [178,223,138],
    [251,154,153],
    [253,191,111],
    [202,178,214],
    ]) / 256.0

def get_hexbin_coords(ax, xlims, ylims, gridsize):
    coords = ax.hexbin([], [], gridsize=gridsize, extent=tuple(xlims)+tuple(ylims)).get_offsets()
    del ax.collections[-1]
    return coords

def plot_transparent_hexbin(ax, func, xlims, ylims, gridsize, color):
    cdict = {'red':   ((0., color[0], color[0]), (1., color[0], color[0])),
             'green': ((0., color[1], color[1]), (1., color[1], color[1])),
             'blue':  ((0., color[2], color[2]), (1., color[2], color[2])),
             'alpha': ((0., 0., 0.), (1., 1., 1.))}

    new_cmap = LinearSegmentedColormap('Custom', cdict)
    plt.register_cmap(cmap=new_cmap)

    coords = get_hexbin_coords(ax, xlims, ylims, gridsize)
    c = func(coords)
    x, y = coords.T

    ax.hexbin(x.ravel(), y.ravel(), c.ravel(),
              cmap=new_cmap, linewidths=0., edgecolors='none',
              gridsize=gridsize, vmin=0., vmax=1., zorder=1)

def decode_density(latent_locations, gamma, decode, weight=1.):
    mu, sigmasq = decode(gamma, latent_locations[:,None,:])

    def density(r):
        distances = np.sqrt(((r[None,:,:] - mu)**2 / sigmasq).sum(2))
        return weight * (scipy.stats.norm.pdf(distances) / np.sqrt(sigmasq).prod(2)).mean(0)

    return density

def reverse_density(y, gamma, decode, weight=1.):

    def density(r):
        mu, sigmasq = decode(gamma, r)
        distances = np.sqrt(((mu - y)**2 / sigmasq).sum(1))
        return weight * (scipy.stats.norm.pdf(distances) / np.sqrt(sigmasq).prod(1))

    return density

def make_plotter(encoder, decoder, global_prior_natparams, params, y):
    T, K, _ = y.shape
    figure, axes = plt.subplots(3, K, figsize=(2*K,6))

    def create_ellipse(mean, cov, sd=2.):
        t = np.linspace(0, 2*np.pi, 100) % (2*np.pi)
        circle = np.vstack((np.sin(t), np.cos(t)))
        return (sd*np.dot(np.linalg.cholesky(cov), circle) + mean[:,None]).T

    def init():
        global_natparams, gamma, phi = params
        global_stats = pgm_expectedstats(global_natparams)

        encoder_out = encoder(phi, y)
        x_encV, x_enc = gaussian.natural_to_standard(
            gaussian.pack_dense(encoder_out[0], encoder_out[1]))

        local_natparams = local_natparams_ep(
            global_prior_natparams, global_natparams, global_stats, encoder_out)
        x_hatV, x_hat = gaussian.natural_to_standard(local_natparams[1])

        decoder_out = decoder(gamma, x_hat)
        y_hat, y_hatV = decoder_out[0], expand_diagonal(decoder_out[1])
        # y_hat = np.mean(decoder(gamma, gaussian.random_samples(encoder_out[1], expand_diagonal(encoder_out[0]), 100))[0], axis=0)
        pi = global_natparams[0][0] / (global_natparams[0][0] + global_natparams[0][1])
        niw_stats = niw.expectedstats(global_natparams[1])
        mu = gaussian.natural_to_standard(niw_stats)[1].reshape(4,2)
        V = gaussian.natural_to_standard(niw_stats)[0].reshape(4,2,2)

        for i in range(K):
            # data space
            axes[0, i].set_ylim(-2,2)
            axes[0, i].set_xlim(-2,2)
            axes[0, i].set_aspect('equal')
            axes[0, i].autoscale(False)
            axes[0, i].plot(y[:,i,0], y[:,i,1], color='k', marker='.', linestyle='', markersize=1)
            axes[0, i].plot(y_hat[:,i,0], y_hat[:,i,1], color='r', marker='.', linestyle='', markersize=4)
            # latent space
            axes[1, i].set_ylim(-2,2)
            axes[1, i].set_xlim(-2,2)
            axes[1, i].set_aspect('equal')
            axes[1, i].autoscale(False)
            axes[1, i].plot(x_hat[:,i,0], x_hat[:,i,1], color='r', marker='.', linestyle='', markersize=2)
            axes[1, i].plot(x_enc[:,i,0], x_enc[:,i,1], color='k', marker='.', linestyle='', markersize=1)

            # gaussian priors
            j = (i + 1) % K
            weights = (1 - pi[i]) * (1 - pi[j]), pi[i] * (1 - pi[j]), (1 - pi[i]) * pi[j], pi[i] * pi[j]
            weights = weights / np.max(weights)
            for (m, v, w) in zip(mu, V, weights):
                latent_ellipse = create_ellipse(m, v)
                data_ellipse = decoder(gamma, latent_ellipse)[0]
                axes[0, i].plot(data_ellipse[:, 0], data_ellipse[:, 1], alpha=w, linestyle='-', linewidth=1)
                axes[1, i].plot(latent_ellipse[:, 0], latent_ellipse[:, 1], alpha=w, linestyle='-', linewidth=1)

            # for (m, v) in zip(y_hat[0:3,i], y_hatV[0:3,i]):
            #     ellipse = create_ellipse(m, v)
            #     axes[0, i].plot(ellipse[:, 0], ellipse[:, 1], alpha=w, linestyle='-', linewidth=1)
            # for (m, v) in zip(x_hat[0:10,i], x_hatV[0:10,i]):
            #     ellipse = create_ellipse(m, v)
            #     axes[1, i].plot(ellipse[:, 0], ellipse[:, 1], alpha=w, linestyle='-', linewidth=1)
            plt.pause(0.1)

        figure.tight_layout()
        plt.pause(.5)

    def plot(iter, val, params, grad):
        print('{}: {}'.format(iter, val))
        if iter % 10 == 0:
            global_natparams, gamma, phi = params
            global_stats = pgm_expectedstats(global_natparams)

            encoder_out = encoder(phi, y)
            x_encV, x_enc = gaussian.natural_to_standard(
                gaussian.pack_dense(encoder_out[0], encoder_out[1]))

            local_natparams = local_natparams_ep(
                global_prior_natparams, global_natparams, global_stats, encoder_out)
            x_hatV, x_hat = gaussian.natural_to_standard(local_natparams[1])

            decoder_out = decoder(gamma, x_hat)
            y_hat, y_hatV = decoder_out[0], expand_diagonal(decoder_out[1])

            pi = global_natparams[0][0] / (global_natparams[0][0] + global_natparams[0][1])
            niw_stats = niw.expectedstats(global_natparams[1])
            mu = gaussian.natural_to_standard(niw_stats)[1].reshape(4,2)
            V = gaussian.natural_to_standard(niw_stats)[0].reshape(4,2,2)

            for i in range(K):
                axes[0, i].lines[1].set_data(y_hat[:,i,0], y_hat[:,i,1])
                axes[1, i].lines[0].set_data(x_hat[:, i, 0], x_hat[:, i, 1])
                axes[1, i].lines[1].set_data(x_enc[:, i, 0], x_enc[:, i, 1])

                if iter % 100 == 0:
                    axes[2,i].clear()
                    axes[2, i].set_ylim(-2,2)
                    axes[2, i].set_xlim(-2,2)
                    axes[2, i].set_aspect('equal')
                    axes[2, i].autoscale(False)
                    axes[2, i].plot(y[:,i,0], y[:,i,1], color='k', marker='.', linestyle='', markersize=1)

                # gaussian priors
                j = (i + 1) % K
                weights = (1 - pi[i]) * (1 - pi[j]), pi[i] * (1 - pi[j]), (1 - pi[i]) * pi[j], pi[i] * pi[j]
                weights = weights / np.max(weights)
                for (k, m, v, w) in zip(range(4), mu, V, weights):
                    latent_ellipse = create_ellipse(m, v)
                    data_ellipse = decoder(gamma, latent_ellipse)[0]
                    axes[0, i].lines[k+2].set_data(data_ellipse[:,0], data_ellipse[:,1])
                    axes[0, i].lines[k+2].set_alpha(w)
                    axes[1, i].lines[k+2].set_data(latent_ellipse[:,0], latent_ellipse[:,1])
                    axes[1, i].lines[k+2].set_alpha(w)

                    # if iter == 200:
                    #     gridsize = 75
                    #     num_samples = 1000
                    #     xlim, ylim = axes[0, i].get_xlim(), axes[0, i].get_ylim()
                    #     samples = npr.RandomState(0).multivariate_normal(m, v, num_samples)
                    #     density = decode_density(samples, gamma, decoder, w)
                    #     plot_transparent_hexbin(axes[0, i], density, xlim, ylim, gridsize, colors[k % len(colors)])

                    if iter % 100 == 0:
                        xlim, ylim = axes[2, i].get_xlim(), axes[2, i].get_ylim()

                        gridsize = 75
                        num_samples = 1000
                        samples = npr.RandomState(0).multivariate_normal(m, v, num_samples)
                        density = decode_density(samples, gamma, decoder, w)
                        plot_transparent_hexbin(axes[2, i], density, xlim, ylim, gridsize, colors[k % len(colors)])

                # for (n, m, v) in zip(range(y_hat.shape[0]), y_hat[0:3,i], y_hatV[0:3,i]):
                #     ellipse = create_ellipse(m, v)
                #     axes[0, i].lines[6+n].set_data(ellipse[:, 0], ellipse[:, 1])

                # for (n, m, v) in zip(range(x_hat.shape[0]), x_hat[0:10,i], x_hatV[0:10,i]):
                #     ellipse = create_ellipse(m, v)
                #     axes[1, i].lines[5+n].set_data(ellipse[:, 0], ellipse[:, 1])

                # if iter % 10 == 0:

                    # gridsize = 75
                    # axes[2,i].clear()
                    # axes[2, i].set_ylim(-2,2)
                    # axes[2, i].set_xlim(-2,2)
                    # axes[2, i].set_aspect('equal')
                    # axes[2, i].autoscale(False)
                    # xlim, ylim = axes[2, i].get_xlim(), axes[2, i].get_ylim()

                    # j = npr.randint(0, y_hat.shape[0])
                    # density = reverse_density(y[j,i], gamma, decoder, 1.)
                    # plot_transparent_hexbin(axes[2, i], density, xlim, ylim, gridsize, colors[k % len(colors)])
                    # foo, bar = gaussian.natural_to_standard(encoder_out[0][j,i], expand_diagonal(encoder_out[1])[j,i])
                    # ellipse = create_ellipse(foo, bar)
                    # axes[2,i].plot(y[j,i,0], y[j,i,1], linestyle='', marker='x')
                    # axes[2,i].plot(foo[0], foo[1], linestyle='', marker='o')
                    # axes[2,i].plot(ellipse[:, 0], ellipse[:, 1], linestyle='-', linewidth=1)

            plt.pause(0.1)

    init()

    return plot

if __name__ == "__main__":
    N = P = 2
    K = 3
    T = 800
    # V = .25*np.array([[2,1],[1,2]])
    npr.seed(0)
    pi, mu, V, y, z = create_gmm_cycle_2d_data(20, 20, K, T)
    V = 2*np.tile(np.eye(N), (2,2,1,1))
    #V = 2*np.tile(np.eye(N), (2,2,1,1))
    J = np.linalg.inv(V)

    # construct recognition and decoder networks and initialize them
    encoder, phi = \
        init_gresnet(P, [(40, np.tanh), (40, np.tanh), (40, np.tanh), (40, np.tanh), (2*N, gaussian_info)])
    decoder, gamma = \
        init_gresnet(N, [(40, np.tanh), (40, np.tanh), (40, np.tanh), (2*P, gaussian_mean)])
    loglike = make_loglike(decoder)

    # global prior and variational posterior natparams
    global_prior_natparams = init_global_natparams(K, N, alpha=0.5/K, niw_conc=100.)
    global_natparams = init_global_natparams(K, N, alpha=1., niw_conc=0.5, random_scale=1.0)

    params = global_natparams, gamma, phi
    # params = pickle.load(open('params.pkl'))

    plot = make_plotter(encoder, decoder, global_prior_natparams, params, y)
    gradfun = make_gradfun(
        local_inference_ep, encoder, loglike, global_prior_natparams, pgm_expectedstats, y)

    params = adam(gradfun(batch_size=50, num_samples=1, natgrad_scale=1e4, callback=plot),
                 params, num_iters=10000, step_size=1e-3)
    pickle.dump(params, open('params_10k.pkl', 'wb'))