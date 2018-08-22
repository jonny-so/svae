from util import add, sub, scale, zeros_like, square, sqrt, div, add_scalar, concat
 
# TODO make optimizers into monads!
# TODO track grad statistics
 
def adam(gradfun, allparams, num_iters, step_size, b1=0.9, b2=0.999, eps=1e-8):
    natparams, params = allparams[:1], allparams[1:]
    m = zeros_like(params)
    v = zeros_like(params)
    i = 0
    accumulate = lambda rho, a, b: add(scale(1-rho, a), scale(rho, b))
 
    for i in xrange(num_iters):
        grad = gradfun(allparams, i)
        natgrad, grad = grad[:1], grad[1:]
 
        m = accumulate(b1, grad, m)          # first moment estimate
        v = accumulate(b2, square(grad), v)  # second moment estimate
        mhat = scale(1./(1 - b1**(i+1)), m)  # bias correction
        vhat = scale(1./(1 - b2**(i+1)), v)
        update = scale(step_size, div(mhat, add_scalar(eps, sqrt(vhat))))
 
        natparams = sub(natparams, scale(step_size, natgrad))
        params = sub(params, update)
        allparams = concat(natparams, params)
 
    return allparams
