#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
log = lib.logger.Logger(sys.stdout, 4)

def print_ply():
    msg = ('Ply format not yet available')
    raise NotImplementedError(msg)

def print_gnu():
    msg = ('Gnuplot format not yet available')
    raise NotImplementedError(msg)

def print_txt(filename, inuc):

    log.info('Surface file is : %s' % (filename))
    idx = 'atom'+str(inuc)
    log.info('Surface info for atom : %d' % inuc)

    with h5py.File(filename) as f:
        i = f[idx+'/inuc'].value
        xnuc = f[idx+'/xnuc'].value
        log.info('Nuclei %d position : %8.5f %8.5f %8.5f', i, *xnuc)
        xyzrho = f[idx+'/xyzrho'].value
        log.info('Nuclei rho %d position : %8.5f %8.5f %8.5f', i, *xyzrho)
        npang = f[idx+'/npang'].value
        log.info('Number of angular points : %d', npang)
        ntrial = f[idx+'/ntrial'].value
        log.info('Ntrial : %d', ntrial)
        rsurf = f[idx+'/rsurf'].value
        nlimsurf = f[idx+'/nlimsurf'].value
        coords = f[idx+'/coords'].value
        rmin = f[idx+'/rmin'].value
        rmax = f[idx+'/rmax'].value
        log.info('Rmin and rmax for surface : %8.5f %8.5f', rmin, rmax)
        surf_file = filename+'.txt'
        with open(surf_file, 'w') as f2:
            for i in range(npang):
                data = str(rsurf[i,:nlimsurf[i]])[1:-1]
                f2.write('%.15f %.15f %.15f %.15f %.15f %s\n' % \
                (coords[i,0],coords[i,1],coords[i,2],coords[i,3],coords[i,4],data))

if __name__ == '__main__':
    name = 'h2o.chk.h5'
    inuc = 0
    print_txt(name,inuc)
