import autograd.numpy as np
import autograd.numpy.random as npr

def pack_dense(a, b):
    assert(a.shape == b.shape)
    return np.stack([a[...,None], b[...,None]], axis=-1)

def unpack_dense(x):
    assert(x.shape[-1] == 2)
    return x[...,0], x[...,1]

def expectedstats(natparams):
    assert(natparams.shape[-1] == 2)
    j = -2 * natparams[..., 0]
    v = 1./j
    m = natparams[..., 1]*v
    return np.stack([v + np.power(v,2), m], axis=-1)

def natural_to_standard(natparams):
    assert(natparams.shape[-1] == 2)
    a, b = unpack_dense(natparams)
    j = -2*a
    return 1./j, b/j

def standard_to_natural(v, m):
    return pack_dense(-.5/v, m/v)

def logZ(natparams):
    v, m = unpack_dense(natparams)
    return .5*np.log(2*np.pi) + .5*np.log(v) + .5*np.power(m, 2)/v
