import numpy as np
import corner as triangle
from sampler import SliceSampler


def lnprob_gaussian(x, ivar):
    return -0.5 * np.sum(ivar * x ** 2)


def rosenbrock(x, a=1, b=100):
    f = ((a - x[0])**2 + b * (x[1] - x[0]**2)**2)
    return -f


def test_gaussian(niter=10000, ndim=2):

    ivar = 1. / np.random.rand(ndim)
    p0 = np.random.rand(ndim)

    Sigma = np.diag(ivar)
    ss = SliceSampler(Sigma, lnprob_gaussian, ivar)

    res = [r for r in ss.sample(p0, lnprob_gaussian(p0, ivar), niter=niter)]
    fig = triangle.corner(ss._chain)
    print('----\nGaussian')
    print('{} likelihood calls for {} iterations'.format(ss.nlike, niter))
    fig.show()


def test_rosenbrock(niter=10000, a=1, b=100):
    p0 = np.array([-1, 1])
    Sigma = np.diag([a, b])
    ss = SliceSampler(Sigma, rosenbrock, a=a, b=b)

    res = [r for r in ss.sample(p0, rosenbrock(p0, a=a, b=b), niter=niter)]
    fig = triangle.corner(ss._chain)
    best = ss._chain[np.argmax(ss._lnprob), :]
    print('----\nRosenbrock')
    print('{} likelihood calls for {} iterations'.format(ss.nlike, niter))
    print('max likelihood={}'.format(best))
    fig.show()
    
if __name__ == "__main__":
    test_gaussian(ndim=4)
    test_rosenbrock()
