#!/usr/bin/env python

import numpy, ctypes
from pyscf import lib
from pyscf.lib import parameters as param

libdft = lib.load_library('libdft')

EPS = 1e-7
LEBEDEV_NGRID = numpy.asarray((
    1   , 6   , 14  , 26  , 38  , 50  , 74  , 86  , 110 , 146 ,
    170 , 194 , 230 , 266 , 302 , 350 , 434 , 590 , 770 , 974 ,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810))

#########################
# JCP 41 3199 (1964).
BRAGG = 1.0/param.BOHR * numpy.array((0,  # Ghost atom
        0.35,                                     1.40,             # 1s
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             # 2s2p
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             # 3s3p
        2.20, 1.80,                                                 # 4s
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, # 3d
                    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,             # 4p
        2.35, 2.00,                                                 # 5s
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, # 4d
                    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,             # 5p
        2.60, 2.15,                                                 # 6s
        1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   # La, Ce-Eu
        1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             # Gd, Tb-Lu
              1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50, # 5d
                    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,             # 6p
        1.80, 2.15,                                                 # 7s
        1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                    1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
        1.75, 1.75,
        1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75))


# from Gerald Knizia's CtDftGrid, which is based on
#       http://en.wikipedia.org/wiki/Covalent_radius
# and
#       Beatriz Cordero, Veronica Gomez, Ana E. Platero-Prats, Marc Reves,
#       Jorge Echeverria, Eduard Cremades, Flavia Barragan and Santiago
#       Alvarez.  Covalent radii revisited. Dalton Trans., 2008, 2832-2838,
#       doi:10.1039/b801115j
COVALENT = 1.0/param.BOHR * numpy.array((0,  # Ghost atom
        0.31,                                     0.28,             # 1s
        1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,             # 2s2p
        1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,             # 3s3p
        2.03, 1.76,                                                 # 4s
        1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22, # 3d
                    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,             # 4p
        2.20, 1.95,                                                 # 5s
        1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, # 4d
                    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,             # 5p
        2.44, 2.15,                                                 # 6s
        2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,                   # La, Ce-Eu
        1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87,             # Gd, Tb-Lu
              1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, # 5d
                    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,             # 6p
        2.60, 2.21,                                                 # 7s
        2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def lebgrid(npang):

    if npang not in LEBEDEV_NGRID:
        raise ValueError('Lebgrid unsupported angular grid %d' % npang)
    else:
        grids = numpy.zeros((npang,4))
        agrids = numpy.zeros((npang,5))
        libdft.MakeAngularGrid(grids.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(npang))

    for i in range(npang):
        agrids[i,4] = 4.0*numpy.pi*grids[i,3]
        rxy = grids[i,0]*grids[i,0] + grids[i,1]*grids[i,1]
        r = numpy.sqrt(rxy + grids[i,2]*grids[i,2])
        if (rxy < EPS):
            if (grids[i,2] >= 0.0):
                agrids[i,0] = +1.0
            else:
                agrids[i,0] = -1.0
            agrids[i,1] = 0.0
            agrids[i,3] = 0.0
            agrids[i,2] = 1.0
        else:
            rxy = numpy.sqrt(rxy)
            agrids[i,0] = grids[i,2]/r
            agrids[i,1] = numpy.sqrt((1.0-agrids[i,0])*(1.0+agrids[i,0]))
            agrids[i,2] = grids[i,0]/rxy
            agrids[i,3] = grids[i,1]/rxy

    return agrids

if __name__ == '__main__':
    npang = 5810
    agrid = lebgrid(npang)
    with open('agrid.txt', 'w') as f2:
        f2.write('# Point 1 2 3 4 weight\n')
        for i in range(npang):
            f2.write('%d   %.6f  %.6f  %.6f  %.6f  %.6f\n' % \
            ((i+1), agrid[i,0], agrid[i,1], agrid[i,2], agrid[i,3], agrid[i,4]))
