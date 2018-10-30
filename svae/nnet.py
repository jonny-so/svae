from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.builtins import tuple as tuple_
from toolz import curry

from util import compose, sigmoid, softmax, log1pexp, isarray, unbox


### util

def rand_partial_isometry(m, n):
    d = max(m, n)
    return np.linalg.qr(npr.randn(d,d))[0][:m,:n]

def identity_isometry(m, n):
    return np.eye(max(m,n))[:m,:n]

def _make_ravelers(input_shape):
    ravel = lambda inputs: np.reshape(inputs, (-1, input_shape[-1]))
    unravel = lambda outputs: np.reshape(outputs, input_shape[:-1] + (-1,))
    return ravel, unravel


### basic layer stuff

layer = curry(lambda nonlin, W, b, inputs: nonlin(np.matmul(inputs, W) + b))
init_layer_random = curry(lambda d_in, d_out, scale:
                          (scale*npr.randn(d_in, d_out), scale*npr.randn(d_out)))
init_layer_glorot = lambda d_in, d_out: \
    (np.sqrt(2./(d_in + d_out))*npr.randn(d_in, d_out), np.zeros(d_out))
init_layer_partial_isometry = lambda d_in, d_out: \
    (rand_partial_isometry(d_in, d_out), npr.randn(d_out))
init_layer = lambda d_in, d_out, fn=init_layer_random(scale=1e-2): fn(d_in, d_out)

### special output layers to produce Gaussian parameters

@curry
def gaussian_mean(inputs, sigmoid_mean=False):
    mu_input, sigmasq_input = np.split(inputs, 2, axis=-1)
    mu = sigmoid(mu_input) if sigmoid_mean else mu_input
    sigmasq = log1pexp(sigmasq_input)
    return tuple_((mu, sigmasq))

@curry
def gaussian_info(inputs):
    J_input, h = np.split(inputs, 2, axis=-1)
    J = -1./2 * log1pexp(J_input)
    return tuple_((J, h))

@curry
def gaussian_info_mix(classes, inputs):
    mlp_logits = inputs[...,-classes:]
    mlp_mixitem_in = np.reshape(inputs[...,:-classes], inputs.shape[:-1] + (classes,2,-1))
    h = mlp_mixitem_in[...,1,:]
    J = -1./2 * log1pexp(mlp_mixitem_in[...,0,:])
    return tuple_((J, h, mlp_logits))

### multi-layer perceptrons (MLPs)

@curry
def _mlp(nonlinearities, params, inputs):
    ravel, unravel = _make_ravelers(inputs.shape)
    eval_mlp = compose(layer(nonlin, W, b)
                       for nonlin, (W, b) in zip(nonlinearities, params))
    out = eval_mlp(ravel(inputs))
    return unravel(out) if isarray(out) else map(unravel, out)

def init_mlp(d_in, layer_specs, **kwargs):
    dims = [d_in] + [l[0] for l in layer_specs]
    nonlinearities = [l[1] for l in layer_specs]
    params = [init_layer(d_in, d_out, *spec[2:])
              for d_in, d_out, spec in zip(dims[:-1], dims[1:], layer_specs)]
    return _mlp(nonlinearities), params


### turn a gaussian_mean MLP into a log likelihood function

def _diagonal_gaussian_loglike(x, mu, sigmasq):
    mu = mu if mu.ndim == (x.ndim+1) else mu[...,None,:]
    K, p = mu.shape[-2:]
    T = np.product(mu.shape[:-2])
    assert x.shape == mu.shape[:-2] + (p,)
    return -T*p/2.*np.log(2*np.pi) + (-1./2*np.sum((x[...,None,:]-mu)**2 / sigmasq)
                                      -1./2*np.sum(np.log(sigmasq))) / K

def make_loglike(gaussian_mlp):
    def loglike(params, inputs, targets):
        return _diagonal_gaussian_loglike(targets, *gaussian_mlp(params, inputs))
    return loglike


### our version of Gaussian resnets

gaussian_mlp_types = {
    gaussian_mean.func: 'mean',
    gaussian_info.func: 'info',
    gaussian_info_mix.func: 'info_mix'}

def gaussian_mlp_type(layer_specs):
    return gaussian_mlp_types[layer_specs[-1][1].func]

@curry
def _gresnet(mlp_type, mlp, params, inputs):
    ravel, unravel = _make_ravelers(inputs.shape)
    mlp_params, (W, b1, b2) = params

    if mlp_type == 'mean':
        mu_mlp, sigmasq_mlp = mlp(mlp_params, inputs)
        mu_res = unravel(np.dot(ravel(inputs), W) + b1)
        return tuple_((mu_mlp + mu_res, sigmasq_mlp))
    elif mlp_type == 'info':
        J_mlp, h_mlp = mlp(mlp_params, inputs)
        J_res = -1./2 * log1pexp(b2)
        h_res = unravel(np.dot(ravel(inputs), W) + b1)
        return tuple_((J_mlp + J_res, h_mlp + h_res))
    elif mlp_type == 'info_mix':
        d_out = b1.shape[-1]
        J_mlp, h_mlp, logits_mlp = mlp(mlp_params, inputs)
        J_mlp = np.reshape(J_mlp, J_mlp.shape[:-1] + (-1,d_out))
        J_res = -1./2 * log1pexp(b2)
        h_mlp = np.reshape(h_mlp, h_mlp.shape[:-1] + (-1,d_out))
        h_res = unravel(np.dot(ravel(inputs), W) + b1)
        return tuple_((J_mlp + J_res, h_mlp + h_res[...,None,:], logits_mlp))

def init_gresnet(d_in, layer_specs, res_init=rand_partial_isometry):
    d_out = layer_specs[-1][0] // 2
    res_params = res_init(d_in, d_out), np.zeros(d_out), np.zeros(d_out)
    mlp, mlp_params = init_mlp(d_in, layer_specs)
    return _gresnet(gaussian_mlp_type(layer_specs), mlp), (mlp_params, res_params)

def init_gresnet_mix(d_in, d_out, classes, layer_specs, res_init=rand_partial_isometry):
    layer_specs = layer_specs + [((2*d_out + 1)*classes, gaussian_info_mix(classes))]
    res_params = res_init(d_in, d_out), np.zeros(d_out), np.zeros(d_out)
    mlp, mlp_params = init_mlp(d_in, layer_specs)
    return _gresnet(gaussian_mlp_type(layer_specs), mlp), (mlp_params, res_params)
