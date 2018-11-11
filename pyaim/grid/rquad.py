#!/usr/bin/env python

import numpy
from pyaim.grid import legendre

def rquad(nr,r0,rfar,rad,iqudr,mapr):

    rmesh = numpy.zeros(nr)
    dvol = numpy.zeros(nr)
    dvoln = numpy.zeros(nr)
 
    #if (rfar-r0 <= 0.001):
    #    raise RuntimeError('rmax < rmin ??') 
    #if (mapr > 3):
    #    raise RuntimeError('not allowed radial mapping')

    # Determine eta parameter in case of radial mapping
    rfarc = rfar - r0
    if (mapr == 1):
        eta = 2.0*rad/rfarc
    elif (mapr == 2):
        eta = 2.0*numpy.exp(-rfarc/rad)/(1.0-numpy.exp(-rfarc/rad))

    #if (iqudr == 1):
    xr, rwei = legendre.legendre(nr)

    # Determine abscissas and volume elements.
    # for finite range (a..b) the transformation is y = (b-a)*x/2+(b+a)/2
    # x = (b-a)*0.5_rp*x+(b+a)*0.5_rp
    # w = w*(b-a)*0.5_rp
    if (mapr == 0):
        for i in range(nr):
            aa = (rfar-r0)/2.0
            bb = (rfar+r0)/2.0
            u = xr[i]
            r = aa*u+bb
            rmesh[i] = r
            dvoln[i] = r*aa
            dvol[i] = dvoln[i]*r
    elif (mapr == 1):
        for i in range(nr):
            u = xr[i]
            den = (1.0-u+eta)
            r = rad*(1.0+u)/den + r0
            rmesh[i] = r
            #if (numpy.abs(den) >= RHOEPS):
            dvoln[i] = rad*(2.0+eta)/den/den*r
            #else
            #dvoln(n) = 0.0_rp
            dvol[i] = dvoln[i]*r
    elif (mapr == 2):
        for i in range(nr):
            u = xr[i]
            den = (1.0-u+eta)
            r = rad*numpy.log((2.0+eta)/den) + r0
            rmesh[i] = r
            dvoln[i] = r*rad/den
            dvol[i] = dvoln[i]*r

    return rmesh, rwei, dvol, dvoln
 
if __name__ == '__main__':
    nr = 10
    x, w = legendre.legendre(nr) 
    r0 = 0
    rfar = 2
    rad = 1.5
    iqudr = 1
    mapr = 1
    rm, rw, dv, dvn = rquad(nr,r0,rfar,rad,iqudr,mapr)
    print rm
    #print rw
    #print dv
    #print dvn
