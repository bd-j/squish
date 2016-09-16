import numpy as np


class SliceSampler(object):

    def __init__(self, Sigma, lnpostfn, **postargs, **postkwargs):
        # Useful matrices for coordinate transforms
        self.Sigma = Sigma
        self.L = np.linalg.cholesky(Sigma)
        self.Linv = np.linalg.inv(self.L)
        
        self._lnpostfn = lnpostfn
        self.postargs = postargs
        self.postkwargs = postkwargs

    def lnpostfn(self, pos, ctype='theta'):
        if ctype == 'theta':
            return self._lnpostfn(pos, *self.postargs, **self.postkwargs)
        elif ctype == 'x':
            return self._lnpostfn(self.x_to_theta(pos), *self.postargs, **self.postkwargs)
    
    def theta_to_x(self, theta):
        """transform from parameter space Theta to the sampling space X
        """
        return np.dot(self.Linv, theta)

    def x_to_theta(self, x):
        return np.dot(self.L, x)
    
    def random_direction(self):
        n = np.random.normal(size=self.ndim)
        n /= magntiude(n)
        return n

    @property
    def ndim(self):
        return self.Sigma.shape[0]

    def one_sample(self, pos0, lnp0=None, step_out=True):

        # We choose unit normal direction vector randomly on the n-sphere
        direction = self.random_direction()
        # And transform into the parameter space, including scaling (i.e. step
        # sizes in each dimension), and then renormalize
        pvector = self.x_to_theta(direction)
        pscale = magnitude(pdirection)
        pdirection /= pscale

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
        x_l = x0 - r * stepsize * direction
        x_r = x0 + (1 - r) * stepsize * direction

        # Step the left and right limits out until you get below the slice probability
        if step_out:
            lnp_l = self.lnpostfn_x(x_l)
            lnp_r = self.lnpostfn_x(x_r)
            while lnp_l > lnp_slice:
                x_l = x_l - stepsize * direction
                lnp_l = self.lnpostfn(x_l, ctype=ctype)

            while lnp_r > lnp_slice:
                x_r = x_r + stepsize * direction
                lnp_r = self.lnpostfn(x_r, ctype=ctype)

        # Now sample within the limits, shrinking limits to new samples until you hit the slice lnp
        while True:
            rr = np.random.uniform()
            xlength = magnitude(x_r - x_l)
            x_try = x_l + rr * xlength * direction
            lnp_try = self.lnpostfn(x_try, ctype=ctype)
            if lnp_try > lnp_slice:
                # Boom!
                return x_try, lnp_try
            else:
                # Now we need to compare the distance from left edge to original point
                # to the distance from the left edge to the trial point.
                # could probably do this by dotting xtry-x0 into direction and checking the sign
                dpos = magnitude(x0 - x_l) 
                if dpos > (rr * xlength):
                    # if distance to original point is larger, then trial point is to the 'left'
                    x_l = x_try
                elif dpos < (rr * xlength):
                    x_r = x_try
                else:
                    raise(RuntimeError, "Slice sampler shrank to original point?")


def magnitude(x):
    return np.sqrt(x**2.sum())
