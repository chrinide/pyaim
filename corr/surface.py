#!/usr/bin/env python

from pyaim import surf

name = 'lih.chk'
natm = 2

surface = surf.BaderSurf(name)
surface.rmaxsurf = 14
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.verbose = 4
surface.epsiscp = 0.220
surface.mstep = 300
surface.npang = 5810
surface.leb = True
surface.corr = True
surface.cas = True
for i in range(natm):
    surface.inuc = i
    surface.kernel()

