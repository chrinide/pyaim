#!/usr/bin/env python

import numpy
from pyscf.lib import logger
from pyaim.surf import field

GRADEPS = 1e-10
RHOEPS = 1e-10
MINSTEP = 1e-4
MAXSTEP = 0.75
SAFETY = 0.8
ENLARGE = 1.2

def checkcp(self, x, rho, gradmod):

    iscp = False
    nuc = -2

    for i in range(self.natm):
        r = numpy.linalg.norm(x-self.coords[i])
        if (r < self.epsiscp):
            iscp = True
            nuc = i
            return iscp, nuc

    if (gradmod <= GRADEPS):
        iscp = True
        if (rho <= RHOEPS): 
            nuc = -1

    return iscp, nuc

def gradrho(self, xpoint, h):

    h0 = h
    niter = 0
    rho, grad, gradmod = field.rhograd(self,xpoint)
    grdt = grad
    grdmodule = gradmod

    while (grdmodule > GRADEPS and niter < self.mstep):
        niter += 1
        ier = 1
        while (ier != 0):
            xtemp = xpoint + h0*grdt
            rho, grad, gradmod = field.rhograd(self,xtemp)
            escalar = numpy.einsum('i,i->',grdt,grad) 
            if (escalar < 0.707):
                if (h0 >= MINSTEP):
                    h0 = h0/2.0
                    ier = 1
                else:
                    ier = 0
            else:
                if (escalar > 0.9): 
                    hproo = h0*ENLARGE
                    if (hproo < h):
                        h0 = hproo
                    else:
                        h0 = h
                    h0 = numpy.minimum(MAXSTEP, h0)
                ier = 0
                xpoint = xtemp
                grdt = grad
                grdmodule = gradmod
            #logger.debug(self,'scalar, step in gradrho %.6f %.6f', escalar, h0)

    #logger.debug(self,'nsteps in gradrho %d', niter)

    return xpoint, grdmodule

