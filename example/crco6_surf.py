#!/usr/bin/env python

import surf

name = 'crco6.chk'
surf = surf.BaderSurf(name)
surf.epsilon = 1e-4
surf.verbose = 4
surf.epsiscp = 0.220
surf.mstep = 100
surf.npang = 5810

natm = 13
for i in range(natm):
    surf.inuc = i
    surf.kernel()

