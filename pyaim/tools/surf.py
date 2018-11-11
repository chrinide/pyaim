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
        xnuc = numpy.zeros(3)
        xnuc = f[idx+'/xnuc'].value
        log.info('Nuclei %d position : %8.5f %8.5f %8.5f', i, *xnuc)
        xyzrho = numpy.zeros(3)
        xyzrho = f[idx+'/xyzrho'].value
        log.info('Nuclei rho %d position : %8.5f %8.5f %8.5f', i, *xyzrho)
        npang = f[idx+'/npang'].value
        log.info('Number of angular points : %d', npang)
        ntrial = f[idx+'/ntrial'].value
        log.info('Ntrial : %d', ntrial)
        rsurf = numpy.zeros((npang,ntrial))
        nlimsurf = numpy.zeros((npang), dtype=numpy.int32)
        coords = numpy.zeros((npang,4))
        rsurf = f[idx+'/surface'].value
        nlimsurf = f[idx+'/intersecs'].value
        coords = f[idx+'/coords'].value
        rmin = f[idx+'/rmin'].value
        rmax = f[idx+'/rmax'].value
        for i in range(npang):
            print (coords[i,0],coords[i,1],coords[i,2],coords[i,3],coords[i,4],rsurf[i,:nlimsurf[i]])

