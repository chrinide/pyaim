#!/usr/bin/env python

import sys
sys.path.append('..')
import surf

natm = 2
name = 'lif.chk'
surface = surf.BaderSurf(name)
surface.epsilon = 1e-5
surface.epsroot = 1e-5
surface.verbose = 4
surface.epsiscp = 0.320
surface.mstep = 500
surface.npang = 5810
surface.leb = True
#surface.nptheta = 121
#surface.npphi = 121
#surface.iqudt = 'legendre'
surface.rmaxsurf = 10.0
surface.step = 0.1
surface.corr = False
for i in range(natm):
    surface.inuc = i
    surface.kernel()

