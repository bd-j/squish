import numpy as np


class SliceSampler(object):

    def __init__(self, Sigma, lnpostfn, *postargs, **postkwargs):
        """
        :param Sigma:
            Covariance matrix describing an n-ellipsoid

        :param lnpostfn:
            The function that returns the posterior probability.

        :param *postargs: (optional)
            Arguments to lnpostfn

        :param **postkwargs:
            Keyword arguments to lnpostfn
        """
        # Useful matrices for coordinate transforms
        self.Sigma = Sigma
        self.L = np.linalg.cholesky(Sigma)
        self.Linv = np.linalg.inv(self.L)
        
        self._lnpostfn = lnpostfn
        self.postargs = postargs
        self.postkwargs = postkwargs

        self.reset()

    def reset(self):
        """Reset the chain to have no elements, and zero-out 'nlike' 
        """
        self._chain = np.empty((0, self.ndim))
        self._lnprob = np.empty((0))
        self.nlike = 0
        
    def lnpostfn(self, pos, ctype='theta'):
        """A wrapper on the userdefined posterior function

        :param pos:
            Position in parameter space.  This can either be in the position
            space or in the actual parameter space, depending on the value of
            `ctype`

        :param ctype: (optional, default: 'theta')
            Switch specifiying whether the `pos` parameter is the coordinates in
            the actual parameter space (ctype='theta') or in the whitened
            sampling space (ctype='x')

        :returns lnp:
             The ln of the (unnormalized) posterior probability at `pos`
        """
        self.nlike += 1
        if ctype == 'theta':
            return self._lnpostfn(pos, *self.postargs, **self.postkwargs)
        elif ctype == 'x':
            return self._lnpostfn(self.x_to_theta(pos), *self.postargs,
                                  **self.postkwargs)
        
    def theta_to_x(self, theta):
        """Transform from parameter space Theta to the sampling space X
        """
        return np.dot(self.Linv, theta)

    def x_to_theta(self, x):
        return np.dot(self.L, x)
    
    def random_direction(self):
        """Generate a vector uniformly sampled from the unit n-sphere.
        """
        n = np.random.normal(size=self.ndim)
        n /= magnitude(n)
        return n

    @property
    def ndim(self):
        return self.Sigma.shape[0]

    def sample(self, p0, lnp0, niter=1, storechain=True):
        p, lnp = p0.copy(), lnp0.copy()
        if storechain:
            # N = int(niter / thin)
            N = niter
            self._chain = np.concatenate((self._chain,
                                          np.zeros((N, self.ndim))), axis=0)
            self._lnprob = np.concatenate((self._lnprob,
                                           np.zeros(N)), axis=0)

        for i in range(niter):
            p, lnp = self.one_sample(p, lnp)
            if storechain:
                self._chain[i, :] = p
                self._lnprob[i] = lnp
            yield p, lnp

    def one_sample(self, pos0, lnp0=None, step_out=True):

        # We choose unit normal direction vector uniformly on the n-sphere
        direction = self.random_direction()

        # And transform into the parameter space, including scaling
        # (i.e. step sizes in each dimension), and then renormalize,
        # keeping the scaling separate.
        pvector = self.x_to_theta(direction)
        pscale = magnitude(pvector)
        pdirection = pvector / pscale

        # Now slice sample along the transformed direction vector, with
        # stepping out given by the length of the full direction vector in each
        # dimension.
        return self.one_sample_x(pos0, lnp0=lnp0, stepsize=pscale,
                                 direction=pdirection, step_out=step_out,
                                 ctype='theta')

    def one_sample_x(self, x0, lnp0=None, stepsize=1.0, direction=None,
                     ctype='x', step_out=True):
        """
        """
        if lnp0 is None:
            lnp0 = self.lnpostfn(x0, ctype=ctype)

        # here is the lnp defining the slice
        lnp_slice = lnp0 + np.log(np.random.rand())

        # Move along the direction vector by a scaled uniform random amount
        r = np.random.rand()
        x_l = x0 - r * stepsize * direction
        x_r = x0 + (1 - r) * stepsize * direction

        # Step the left and right limits out until you get below the slice probability
        if step_out:
            lnp_l = self.lnpostfn(x_l, ctype=ctype)
            lnp_r = self.lnpostfn(x_r, ctype=ctype)
            while lnp_l > lnp_slice:
                x_l = x_l - stepsize * direction
                lnp_l = self.lnpostfn(x_l, ctype=ctype)

            while lnp_r > lnp_slice:
                x_r = x_r + stepsize * direction
                lnp_r = self.lnpostfn(x_r, ctype=ctype)

        # Now sample within the limits, shrinking limits to new samples until
        # you hit the slice lnp
        while True:
            rr = np.random.uniform()
            xlength = magnitude(x_r - x_l)
            x_try = x_l + rr * xlength * direction
            lnp_try = self.lnpostfn(x_try, ctype=ctype)
            if lnp_try > lnp_slice:
                # Boom!
                return x_try, lnp_try
            else:
                # Now we need to see if the new point is to the 'right' or
                # 'left' of the original point.  We do this by dotting xtry-x0
                # into direction and checking the sign.
                # dpos = magnitude(x0 - x_l)
                s = np.dot(x_try - x0, direction)
                if s < 0:
                    # if distance to original point is larger, then trial point is to the 'left'
                    x_l = x_try
                elif s > 0:
                    x_r = x_try
                else:
                    raise(RuntimeError, "Slice sampler shrank to original point?")


def magnitude(x):
    return np.sqrt(np.dot(x, x))
