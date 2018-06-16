#!/usr/bin/env python

import h5py
import numpy
import legendre
from pyscf import dft, lib


xnuc = numpy.zeros(3)
xyzrho = numpy.zeros(3)

with h5py.File('h2o.h5') as f:
    rmin = f['atom0/rmin'].value
    rmax = f['atom0/rmax'].value
    xyzrho = f['atom0/xyzrho'].value
    xnuc = f['atom0/xnuc'].value
    npang = f['atom0/npang'].value
    ntrial = f['atom0/ntrial'].value
    rsurf = numpy.zeros((npang,ntrial))
    nlimsurf = numpy.zeros((npang))
    coords = numpy.zeros((npang,4))
    rsurf = f['atom0/surface'].value
    nlimsurf = f['atom0/intersecs'].value
    coords = f['atom0/coords'].value

EPS = 1e-6
def inbasin(r,j):

    isin = False
    rs1 = 0.0
    irange = nlimsurf[j]
    irange = int(irange)
    for k in range(irange):
        rs2 = rsurf[j,k]
        if (r >= rs1-EPS and r <= rs2+EPS):
            if (((k+1)%2) == 0):
                isin = False
            else:
                isin = True
            return isin
        rs1 = rs2

    return isin

betafac = 0.1
nrad = 21
iqudr = 1
brad = rmin*betafac
r0 = brad
print brad
rfar = rmax
rad = 0.1
mapr = 0
rmesh, rwei, dvol, dvoln = legendre.rquad(nrad,r0,rfar,rad,iqudr,mapr)

xcoor = numpy.zeros(3)
for n in range(nrad):
    r = rmesh[n]
    for j in range(npang):
        inside = True
        inside = inbasin(r,j)
        if (inside == True):
            cost = coords[j,0]
            sintcosp = coords[j,1]*coords[j,2]
            sintsinp = coords[j,1]*coords[j,3]
            xcoor[0] = r*sintcosp
            xcoor[1] = r*sintsinp
            xcoor[2] = r*cost    
            p = xnuc + xcoor
            print '%16.8f %16.8f %16.8f %16.8f' % (p[0], p[1], p[2], coords[j,4]*rwei[n])

