#!/usr/bin/env python

import numpy
import ctypes
from pyscf import lib
from pyscf.lib import parameters as param

libdft = lib.load_library('libdft')

HMINIMAL = numpy.finfo(numpy.float64).eps

EPS = 1e-7
LEBEDEV_NGRID = numpy.asarray((
    1   , 6   , 14  , 26  , 38  , 50  , 74  , 86  , 110 , 146 ,
    170 , 194 , 230 , 266 , 302 , 350 , 434 , 590 , 770 , 974 ,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810))

#########################
# JCP 41 3199 (1964).
BRAGG = 1.0/param.BOHR * numpy.array((0,  # Ghost atom
        0.35,                                     1.40,             # 1s
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             # 2s2p
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             # 3s3p
        2.20, 1.80,                                                 # 4s
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, # 3d
                    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             # 4p
        2.35, 2.00,                                                 # 5s
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, # 4d
                    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             # 5p
        2.60, 2.15,                                                 # 6s
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   # La, Ce-Eu
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             # Gd, Tb-Lu
              1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, # 5d
                    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             # 6p
        1.80, 2.15,                                                 # 7s
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                    1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75))


# from Gerald Knizia's CtDftGrid, which is based on
#       http://en.wikipedia.org/wiki/Covalent_radius
# and
#       Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves,
#       Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago
#       Alvarez.  Covalent radii revisited. Dalton Trans., 2008, 2832-2838,
#       doi:10.1039/b801115j
COVALENT = 1.0/param.BOHR * numpy.array((0,  # Ghost atom
        0.31,                                     0.28,             # 1s
        1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,             # 2s2p
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             # 3s3p
        2.03, 1.76,                                                 # 4s
        1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22, # 3d
                    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,             # 4p
        2.20, 1.95,                                                 # 5s
        1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, # 4d
                    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,             # 5p
        2.44, 2.15,                                                 # 6s
        2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                   # La, Ce-Eu
        1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             # Gd, Tb-Lu
              1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, # 5d
                    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,             # 6p
        2.60, 2.21,                                                 # 7s
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def legendre(n):

    x = numpy.zeros(n)
    w = numpy.zeros(n)

    e1 = n*(n+1)
    m = ((n+1)//2)
    for i in range(1,m+1):
        mp1mi = m+1-i
        t = float(4*i-1)*numpy.pi/float(4*n+2)
        x0 = numpy.cos(t)*(1.0-(1.0-1.0/float(n))/float(8*n*n))
        pkm1 = 1.0
        pk = x0
        for k in range(2,n+1):
            pkp1 = 2.0*x0*pk-pkm1-(x0*pk-pkm1)/float(k)
            pkm1 = pk
            pk = pkp1
        d1 = float(n)*(pkm1-x0*pk)
        dpn = d1/(1.0-x0*x0)
        d2pn = (2.0*x0*dpn-e1*pk)/(1.0-x0*x0)
        d3pn = (4.0*x0*d2pn+(2.0-e1)*dpn)/(1.0-x0*x0)
        d4pn = (6.0*x0*d3pn+(6.0-e1)*d2pn)/(1.0-x0*x0)
        u = pk/dpn
        v = d2pn/dpn
        h = -u*(1.0+0.5*u*(v+u*(v*v-d3pn/(3.0*dpn))))
        p = pk+h*(dpn+0.5*h*(d2pn+h/3.0*(d3pn+0.25*h*d4pn)))
        dp = dpn+h*(d2pn+0.5*h*(d3pn+h*d4pn/3.0))
        h = h-p/dp
        xtemp = x0+h
        x[mp1mi-1] = xtemp
        fx = d1-h*e1*(pk+0.5*h*(dpn+h/3.0*(d2pn+0.25*h*(d3pn+0.2*h*d4pn))))
        w[mp1mi-1] = 2.0*(1.0-xtemp*xtemp)/fx/fx

    if ((n%2) == 1):
        x[0] = 0.0
    nmove = ((n+1)//2)
    ncopy = n-nmove
    for i in range(1,nmove+1):
        iback = n+1-i
        x[iback-1] = x[iback-ncopy-1]
        w[iback-1] = w[iback-ncopy-1]
    for i in range(1,n-nmove+1):
        x[i-1] = -x[n-i]
        w[i-1] = w[n-i]

    return x, w

def chebyshev1(n):

  x = numpy.zeros(n)
  w = numpy.zeros(n)

  if (n == 1):
    x[0] = 0.0
    w[0] = numpy.pi
  else:
    for i in range(n):
      angle = float(i)*numpy.pi/float(n-1)
      x[i] = numpy.cos(angle)
      w[i] = numpy.pi/float(n-1)
    w[0] = 0.5*w[0]
    w[n-1] = 0.5*w[n-1]

  return x[::-1], w[::-1]

def chebyshev2(n):

  x = numpy.zeros(n)
  w = numpy.zeros(n)

  for i in range(n):
    angle = numpy.pi*float(n-i)/float(n+1)
    w[i] = numpy.pi/float(n+1)*(np.sin(angle))**2
    x[i] = numpy.cos(angle)

  return x, w

def compositetrap(a,b,n):

    x = numpy.zeros(n)
    w = numpy.zeros(n)
    h = (b-a)/float(n)
    x[0] = a
    x[n-1] = b
    w[0] = h/2.0
    w[n-1] = w[0]
    for i in range(1,n-1):
        w[i] = h
        x[i] = float(i)*h

    return x,w

# Change the -1,1 limits for quadratures
def changelimits(a,b,x,w):  

    nr = len(x)
    for i in range(nr):
        aa = (b-a)/2.0
        bb = (b+a)/2.0
        r = aa*x[i]+bb
        #r = a + aa*(x[i]+1.0)
        x[i] = r
        w[i] = w[i]*aa

    return x,w

def anggrid(iqudt,nptheta,npphi):

    npang = nptheta*npphi
    agrids = numpy.zeros((npang,5))
    delphi = 2.0*numpy.pi/float(npphi)
        
    if (iqudt == 1):
        x, w = legendre(nptheta)
    elif (iqudt == 2):
        x, w = chebyshev1(nptheta)
    else:
        raise NotImplementedError("anggrid only legendre or cheby1")

    tnpang = 0
    for ip in range(npphi):
       phi = ip*delphi
       for it in range(nptheta):
          thang = x[it]
          agrids[tnpang,0] = thang
          agrids[tnpang,1] = numpy.sqrt(1.0-thang*thang)
          agrids[tnpang,2] = numpy.cos(phi)
          agrids[tnpang,3] = numpy.sin(phi)
          agrids[tnpang,4] = w[it]*delphi
          tnpang += 1

    return agrids

def lebgrid(npang):

    if npang not in LEBEDEV_NGRID:
        raise ValueError('Lebgrid unsupported angular grid %d' % npang)
    else:
        grids = numpy.zeros((npang,4))
        agrids = numpy.zeros((npang,5))
        libdft.MakeAngularGrid(grids.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(npang))

    for i in range(npang):
        agrids[i,4] = 4.0*numpy.pi*grids[i,3]
        rxy = grids[i,0]*grids[i,0] + grids[i,1]*grids[i,1]
        r = numpy.sqrt(rxy + grids[i,2]*grids[i,2])
        if (rxy < EPS):
            if (grids[i,2] >= 0.0):
                agrids[i,0] = +1.0
            else:
                agrids[i,0] = -1.0
            agrids[i,1] = 0.0
            agrids[i,3] = 0.0
            agrids[i,2] = 1.0
        else:
            rxy = numpy.sqrt(rxy)
            agrids[i,0] = grids[i,2]/r
            agrids[i,1] = numpy.sqrt((1.0-agrids[i,0])*(1.0+agrids[i,0]))
            agrids[i,2] = grids[i,0]/rxy
            agrids[i,3] = grids[i,1]/rxy

    return agrids

def rquad(nr,r0,rfar,rad,iqudr,mapr):

    rmesh = numpy.zeros(nr)
    dvol = numpy.zeros(nr)
    dvoln = numpy.zeros(nr)
 
    if (rfar-r0 <= 0.001):
        raise RuntimeError('rmax < rmin ??') 

    # Determine eta parameter in case of radial mapping
    rfarc = rfar - r0
    if (mapr == 1):
        eta = 2.0*rad/rfarc
    elif (mapr == 2):
        eta = 2.0*numpy.exp(-rfarc/rad)/(1.0-numpy.exp(-rfarc/rad))
    elif (mapr == 0):
        eta = 0.0
    else:    
        raise NotImplementedError('Only becke or exp mapping available') 

    if (iqudr == 1):
        xr, rwei = legendre(nr)
    else:    
        raise NotImplementedError('Only legendre quadrature available') 

    # Determine abscissas and volume elements.
    # for finite range (a..b) the transformation is y = (b-a)*x/2+(b+a)/2
    # x = (b-a)*0.5*x+(b+a)*0.5
    # w = w*(b-a)*0.5
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
            if (numpy.abs(den) >= EPS):
                dvoln[i] = rad*(2.0+eta)/den/den*r
            else:
                dvoln[i] = 0.0
            dvol[i] = dvoln[i]*r
    elif (mapr == 2):
        for i in range(nr):
            u = xr[i]
            den = (1.0-u+eta)
            r = rad*numpy.log((2.0+eta)/den) + r0
            rmesh[i] = r
            if (numpy.abs(den) >= EPS):
                dvoln[i] = r*rad/den
            else:
                dvoln[i] = 0.0
            dvol[i] = dvoln[i]*r

    return rmesh, rwei, dvol, dvoln

if __name__ == '__main__':
    nr = 10
    npang = 5810
    agrid = lebgrid(npang)
    with open('agrid.txt', 'w') as f2:
        f2.write('# Point 1 2 3 4 weight\n')
        for i in range(npang):
            f2.write('%d   %.6f  %.6f  %.6f  %.6f  %.6f\n' % \
            ((i+1), agrid[i,0], agrid[i,1], agrid[i,2], agrid[i,3], agrid[i,4]))

    iqudt = 1
    nptheta = 101
    npphi = 101
    agrid = anggrid(iqudt,nptheta,npphi) 
    with open('anggrid.txt', 'w') as f2:
        f2.write('# Point 1 2 3 4 weight\n')
        for i in range(nptheta*npphi):
            f2.write('%d   %.6f  %.6f  %.6f  %.6f  %.6f\n' % \
            ((i+1), agrid[i,0], agrid[i,1], agrid[i,2], agrid[i,3], agrid[i,4]))
    
    x, w = chebyshev1(nr) 
    print('Cheb points %s' % x)
    print('Cheb weigths %s' % w)

    a = 0
    b = 2*numpy.pi
    x, w = changelimits(a,b,x,w)   
    print('Cheb points %s' % x)
    print('Cheb weigths %s' % w)

    x, w = compositetrap(a,b,nr)
    print('Trap points %s' % x)
    print('Trap weigths %s' % w)

    nr = 10
    x, w = legendre(nr) 
    print('Legendre points %s' % x)
    print('Legendre weigths %s' % w)

    r0 = 0
    rfar = 2
    rad = 1.5
    iqudr = 1
    mapr = 1
    rm, rw, dv, dvn = rquad(nr,r0,rfar,rad,iqudr,mapr)

