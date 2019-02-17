#!/usr/bin/env python

from pyaim.gamma import surf

name = 'gamma_hf.chk'
natm = 2

surface = surf.BaderSurf(name)
surface.rmaxsurf = 10
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.dimension = 1
surface.verbose = 4
surface.epsiscp = 0.220
surface.mstep = 300
surface.npang = 5810
surface.leb = True
surface.corr = False
surface.cas = False
for i in range(natm):
    surface.inuc = i
    surface.kernel()

