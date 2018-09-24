import autograd.numpy as np
import autograd.numpy.random as npr
import scipy.stats

def gen_data(N, T, tau, scale):
    x = np.cumsum(npr.randn(N,T)*np.sqrt(1./tau), axis=1)
    return np.array(npr.rand(*x.shape) > scipy.stats.norm.cdf(x, scale=scale), dtype=np.int32)

if __name__ == "__main__":
    N = 1000
    T = 100
    tau = 1/(1.0/100)
    y = gen_data(N, T, tau, scale=1.0)
    print(y)