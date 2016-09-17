import numpy as np

from sampler import SliceSampler


def lnprob(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)


ndim = 2
ivar = 1. / np.random.rand(ndim)
p0 = np.random.rand(ndim)

Sigma = np.diag(ivar)
ss = SliceSampler(Sigma, lnprob, ivar)

res = [r for r in ss.sample(p0, lnprob(p0, ivar), niter=10000)]
