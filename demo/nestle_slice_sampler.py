import numpy as np
from nestle import SingleEllipsoidSampler#, MultiEllipsoidSampler
from squish import SliceSampler


class SingleEllipsoidalSliceSampler(SingleEllipsoidalSampler):

    def __init__(self):
        # include all other init stuff.  Or put this in self.set_options()
        self.slicer = SliceSampler(self.ell.axes, self.lnpost)

    def lnpost(self, v):
        return self.loglikelihood(self.prior_transform(v))
         
    def new_point(self, loglstar, niter=10):
        
        ncall = 0
        logl = -float('inf')
        while logl < loglstar:
            while True:
                # Multi ellipsoid version.  bounding_ellipsoids() needs to also
                # return the index of the ellipse it chose.
                #u, iell = bounding_ellipsoids(self.ells, rstate=self.rstate)
                #self.slicer.transform = self.ells[iell].axes
                
                u = self.ell.sample(rstate=self.rstate)
                self.slicer.transform = self.ells.axes
                if np.all(u > 0.) and np.all(u < 1.):
                    break
            self.slicer.reset()
            u, logl = self.slicer.sample(u, None, niter=niter, storechain=False)
            v = self.prior_transform(u)
            ncall += slicer.nlike

        return u, v, logl, ncall
