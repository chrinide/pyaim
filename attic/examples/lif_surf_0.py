#!/usr/bin/env python

from pyaim import surf

name = 'lif.chk'
surf = surf.BaderSurf(name)
surf.epsilon = 1e-4
surf.verbose = 4
surf.epsiscp = 0.220
surf.mstep = 200
surf.inuc = 0
surf.npang = 5810
surf.kernel()

for i in range(surf.npang):
    print "*",i,surf.grids[i,0],surf.grids[i,1],\
                surf.grids[i,2],surf.grids[i,3],\
                surf.nlimsurf[i],surf.rsurf[i,:surf.nlimsurf[i]]
