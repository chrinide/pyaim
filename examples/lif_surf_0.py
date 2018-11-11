#!/usr/bin/env python

from pyaim import surf

name = 'lif.chk'
surf = surf.BaderSurf(name)
surf.epsilon = 1e-4
surf.verbose = 4
surf.epsiscp = 0.180
surf.mstep = 100
surf.inuc = 0
surf.npang = 6
surf.kernel()

