#!/usr/bin/env python

import surf

name = 'cf2.chk'
surf = surf.BaderSurf(name)
surf.epsilon = 1e-5
surf.epssurf = 1e-5
surf.verbose = 4
surf.epsiscp = 0.220
surf.mstep = 100
surf.npang = 5810

surf.inuc = 0
surf.kernel()
surf.inuc = 1
surf.kernel()
surf.inuc = 2
surf.kernel()

