import numpy as np
from copy import deepcopy


class SliceSampler(object):

    def __init__(self, transform, lnpostfn, store_inv=False,
                 postargs=[], postkwargs={}):
        """
        :param transform:
            Transformation matrix from unit n-sphere to parameter space.

        :param lnpostfn:
            The function that returns the posterior probability.

        :param *postargs: (optional)
            Arguments to lnpostfn

        :param **postkwargs:
            Keyword arguments to lnpostfn
        """
        # Useful matrices for coordinate transforms
        self.transform = transform
        if store_inv:
            self.inverse_transform = np.linalg.inv(transform)
        
        self._lnpostfn = lnpostfn
        self.postargs = postargs
        self.postkwargs = postkwargs

        self.reset()

    def reset(self):
        """Reset the chain to have no elements, and zero-out 'nlike' 
        """
        self.chain = np.empty((0, self.ndim))
        self.lnprob = np.empty((0))
        self.nlike = 0
        
    def lnpostfn(self, pos):
        """A wrapper on the userdefined posterior function

        :param pos:
            Position in parameter space.

        :returns lnp:
             The ln of the (unnormalized) posterior probability at `pos`
        """
        self.nlike += 1
        return self._lnpostfn(pos, *self.postargs, **self.postkwargs)
        
    def theta_to_x(self, theta):
        """Transform from parameter space Theta to the sampling space X
        """
        try:
            return np.dot(self.inverse_transform, theta)
        except:
            return np.dot(np.linalg.inv(self.transform), theta)

    def x_to_theta(self, x):
        return np.dot(self.transform, x)

    @property
    def ndim(self):
        return self.transform.shape[0]

    def sample(self, p0, lnp0=None, niter=1, storechain=True):
        """Draw `niter` slice samples.
        """
        p, lnp = p0.copy(), deepcopy(lnp0)
        if storechain:
            # N = int(niter / thin)
            N = niter
            self.chain = np.concatenate((self.chain, np.zeros((N, self.ndim))), axis=0)
            self.lnprob = np.concatenate((self.lnprob, np.zeros(N)), axis=0)

        for i in range(niter):
            p, lnp = self.draw_new_sample(p, lnp)
            if storechain:
                self.chain[i, :] = p
                self.lnprob[i] = lnp
            yield p, lnp

    def draw_new_sample(self, pos0, lnp0=None, step_out=True):

        # We choose unit normal direction vector uniformly on the n-sphere
        nhat = randir(self.ndim)

        # And transform into the parameter space, including scaling
        # (i.e. step sizes in each dimension), and then renormalize,
        # keeping the scaling separate.
        slice_vector = self.x_to_theta(nhat)
        scale = magnitude(slice_vector)
        nhat_slice = slice_vector / scale

        # Now slice sample along the transformed direction vector, with
        # steps out given by the length of the full direction vector in each
        # dimension.
        return self.slice_sample(pos0, lnp0=lnp0, step_out=step_out,
                                 stepsize=scale, nhat=nhat_slice)

    def slice_sample(self, x0, lnp0=None, stepsize=1.0, nhat=None,
                    step_out=True):
        """Draw a slice sample along the unit vector given by `nhat`.
        """
        if lnp0 is None:
            lnp0 = self.lnpostfn(x0)

        # here is the lnp defining the slice
        lnp_slice = lnp0 + np.log(np.random.rand())

        # Move along the direction vector by a scaled uniform random amount
        r = np.random.rand()
        x_l = x0 - r * stepsize * nhat
        x_r = x0 + (1 - r) * stepsize * nhat

        # Step the left and right limits out until you get below the slice probability
        if step_out:
            lnp_l = self.lnpostfn(x_l)
            lnp_r = self.lnpostfn(x_r)
            while lnp_l > lnp_slice:
                x_l = x_l - stepsize * nhat
                lnp_l = self.lnpostfn(x_l)

            while lnp_r > lnp_slice:
                x_r = x_r + stepsize * nhat
                lnp_r = self.lnpostfn(x_r)

        # Now sample within the limits, shrinking limits to new samples until
        # you hit the slice lnp
        while True:
            rr = np.random.uniform()
            xlength = magnitude(x_r - x_l)
            x_try = x_l + rr * xlength * nhat
            lnp_try = self.lnpostfn(x_try)
            if lnp_try > lnp_slice:
                # Boom!
                return x_try, lnp_try
            else:
                # Now we need to see if the new point is to the 'right' or
                # 'left' of the original point.  We do this by dotting xtry-x0
                # into direction and checking the sign.
                s = np.dot(x_try - x0, nhat)
                if s < 0:
                    # if distance to original point is larger, then trial point is to the 'left'
                    x_l = x_try
                elif s > 0:
                    x_r = x_try
                else:
                    raise(RuntimeError, "Slice sampler shrank to original point?")


def randir(n):
    """Generate a vector uniformly sampled from the unit n-sphere.
    """
    nhat = np.random.normal(size=n)
    nhat /= magnitude(nhat)
    return nhat


def magnitude(x):
    return np.sqrt(np.dot(x, x))


class SliceSamplerV2(SliceSampler):
    """A version of the slice sampler that moves on the unit n-sphere but
    transforms the coordinates in this space to real parameter coordinates as
    lnpostfn is called.
    """

    def lnpostfn(self, pos):
        """A wrapper on the userdefined posterior function

        :param pos:
            Position in sampling space.  This will be transformed to the
            parameter space before being passed to the user supplied posterior
            function.

        :returns lnp:
             The ln of the (unnormalized) posterior probability at `pos`
        """
        self.nlike += 1
        return self._lnpostfn(self.x_to_theta(pos), *self.postargs,
                              **self.postkwargs)

    def draw_new_sample(self, pos0, lnp0=None, step_out=True):

        # We choose unit normal direction vector uniformly on the n-sphere
        nhat = randir(self.ndim)
        stepsize = 1.0

        # Now slice sample along the transformed direction vector, with
        # steps out of 1.  Transformation will happen just before lnp is calculated)
        return self.slice_sample(pos0, lnp0=lnp0, step_out=step_out,
                                 stepsize=stepsize, nhat=nhat)
