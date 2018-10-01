#!/usr/bin/env python

import numpy, ctypes
from pyscf import lib

libdft = lib.load_library('libdft')

EPS = 1e-7
LEBEDEV_NGRID = numpy.asarray((
    1   , 6   , 14  , 26  , 38  , 50  , 74  , 86  , 110 , 146 ,
    170 , 194 , 230 , 266 , 302 , 350 , 434 , 590 , 770 , 974 ,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810))

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
