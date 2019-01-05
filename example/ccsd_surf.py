#!/usr/bin/env python

import sys
sys.path.append('..')
import surf

natm = 3
name = 'cf2.chk'
surface = surf.BaderSurf(name)
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.verbose = 4
surface.epsiscp = 0.220
surface.mstep = 200
surface.npang = 5810
surface.leb = False
surface.nptheta = 90
surface.npphi = 180
surface.iqudt = 'legendre'
surface.rmaxsurf = 10.0
surface.step = 0.1
surface.corr = True
surface.occdrop = 1e-6
for i in range(natm):
    surface.inuc = i
    surface.kernel()

