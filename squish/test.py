import numpy as np
import triangle
from sampler import SliceSampler


def lnprob_gaussian(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)


def rosenbrock(x, a=1, b=100):
    f = ((a - x[0])**2 + b * (x[1] - x[0]**2)**2)
    return -f


def test_gaussian():

    ndim = 2
    ivar = 1. / np.random.rand(ndim)
    p0 = np.random.rand(ndim)

    Sigma = np.diag(ivar)
    ss = SliceSampler(Sigma, lnprob_gaussian, ivar)

    res = [r for r in ss.sample(p0, lnprob_gaussian(p0, ivar), niter=10000)]
    fig = triangle.corner(ss._chain())
    fig.show()


def test_rosenbrock():
    p0 = np.array([-1, 1])
    Sigma = np.diag([10, 10])
    ss = SliceSampler(Sigma, rosenbrock, a=1, b=100)

    res = [r for r in ss.sample(p0, rosenbrock(p0, a=1, b=100), niter=10000)]
    fig = triangle.corner(ss._chain)
    best = ss._chain[np.argmax(ss._lnprob), :]
    print(best)
    fig.show()
    
