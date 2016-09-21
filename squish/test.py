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
    lnp0 = lnprob_gaussian(p0, ivar)

    transform = np.linalg.cholesky(np.diag(ivar))
    ss = SliceSampler(transform, lnprob_gaussian, postargs=[ivar])

    res = [r for r in ss.sample(p0, lnp0, niter=niter)]
    fig = triangle.corner(ss.chain)
    print('----\nGaussian')
    print('{} likelihood calls for {} iterations'.format(ss.nlike, niter))
    fig.show()
    return ss


def test_rosenbrock(niter=10000, a=1, b=100, save=False):
    p0 = np.array([-1, 1])
    lnp0 = rosenbrock(p0, a=a, b=b)

    transform = np.linalg.cholesky(np.diag([1, 1]))
    ss = SliceSampler(transform, rosenbrock, postkwargs={'a':a, 'b':b})

    res = [r for r in ss.sample(p0, lnp0, niter=niter)]
    #ss.reset()
    #ss.sample(p0, lnp0, niter=niter, storechain=True)

    print('finished sampling')
    fig = triangle.corner(ss.chain[5000:, :], truths=[a, a], labels=[r'$x_1$', r'$x_2$'])
    free = np.array(fig.axes)[~np.array([ax.has_data() for ax in fig.axes])]
    best = ss.chain[np.argmax(ss.lnprob), :]
    text = 'Rosenbrock ($a={:4.1f}$, $b={:4.1f})$\n'.format(a, b)
    text += '{} iterations\n'.format(niter)
    text += '{} likelihood calls\n'.format(ss.nlike)
    text += 'MAP: {:4.2f}, {:4.2f}'.format(*best)
    free[0].text(0.01, 0.8, text, transform=free[0].transAxes, verticalalignment='top')
    fig.show()
    if save:
        fig.savefig('rosenbrock.png')
    return ss

    
if __name__ == "__main__":
    import matplotlib.pyplot as pl
    pl.rcParams['xtick.direction'] = 'in'
    pl.rcParams['xtick.direction'] = 'in'
    #test_gaussian(ndim=4)
    ss = test_rosenbrock()

    ss = test_rosenbrock(niter=int(1e6))
    import acor
    lag = 10 **(np.arange(4)+1)
    tau1 = [acor.acor(ss.chain[:,1], maxlag=l)[0] for l in lag]
    tau0 = [acor.acor(ss.chain[:,0], maxlag=l)[0] for l in lag]
    for l, t1, t0 in zip(lag, tau1, tau0):
        print('maxlag:{}, tau0:{}, tau1:{}'.format(l, t0, t1))
