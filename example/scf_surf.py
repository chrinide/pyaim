#!/usr/bin/env python

import sys
sys.path.append('..')
import surf

natm = 6
name = 'c2f4.chk'
surface = surf.BaderSurf(name)
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.verbose = 4
surface.epsiscp = 0.220
surface.mstep = 200
surface.npang = 5810
surface.leb = False
surface.nptheta = 121
surface.npphi = 121
surface.iqudt = 'legendre'
surface.rmaxsurf = 10.0
surface.step = 0.1
surface.corr = False
for i in range(natm):
    surface.inuc = i
    surface.kernel()

