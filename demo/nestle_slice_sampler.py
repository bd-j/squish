import numpy as np
from nestle import SingleEllipsoidSampler#, MultiEllipsoidSampler
from squish import SliceSampler


class SingleEllipsoidalSliceSampler(SingleEllipsoidalSampler):

    def __init__(self):
        
        self.slicer = SliceSampler(self.ell.axes, self.lnpost)

    def lnpost(self, v):
        return self.loglikelihood(self.prior_transform(v))
         
    def new_point(self, loglstar, niter=10):
        
        ncall = 0
        logl = -float('inf')
        while logl < loglstar:
            while True:
                # Multi ellipsoid version
                #u, iell = bounding_ellipsoids(self.ells, rstate=self.rstate)
                #slicer.transform = self.ells[iell].axes
                
                u = self.ell.sample(rstate=self.rstate)
                slicer.transform = self.ells.axes
                if np.all(u > 0.) and np.all(u < 1.):
                    break
            slicer.reset()
            u, logl = slicer.sample(u, None, niter=niter, storechain=False)
            v = self.prior_transform(u)
            ncall += slicer.nlike

        return u, v, logl, ncall
