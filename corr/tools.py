#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)

NPROPS = 3
PROPS = ['density', 'kinetic', 'laplacian']
NCOL = 15
DIGITS = 5

def print_surface_ply():
    msg = ('Ply format not yet available')
    raise NotImplementedError(msg)

def print_surface_gnu():
    msg = ('Gnuplot format not yet available')
    raise NotImplementedError(msg)

def print_surface_txt(filename, inuc):

    log.info('Surface file is : %s' % (filename))
    idx = 'atom'+str(inuc)
    log.info('Surface info for atom : %d' % inuc)

    with h5py.File(filename) as f:
        i = f[idx+'/inuc'].value
        xnuc = f[idx+'/xnuc'].value
        log.info('Nuclei %d position : %8.5f %8.5f %8.5f', i, *xnuc)
        xyzrho = f[idx+'/xyzrho'].value
        log.info('Nuclei rho %d position : %8.5f %8.5f %8.5f', i, *xyzrho[i])
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
        surf_file = filename+'_'+idx+'.txt'
        with open(surf_file, 'w') as f2:
            for i in range(npang):
                data = str(rsurf[i,:nlimsurf[i]])[1:-1]
                f2.write('%.15f %.15f %.15f %.15f %.15f %s\n' % \
                (coords[i,0],coords[i,1],coords[i,2],coords[i,3],coords[i,4],data))

def print_properties(name, natm, nmo):                
    aom = numpy.zeros((natm,nmo,nmo))
    totaom = numpy.zeros((nmo,nmo))
    props = numpy.zeros((natm,NPROPS))
    totprops = numpy.zeros((NPROPS))
    with h5py.File(name) as f:
        for i in range(natm):
            idx = 'atom_props'+str(i)
            props[i] = f[idx+'/totprops'].value
            idx = 'ovlp'+str(i)
            aom[i] = f[idx+'/aom'].value
    for i in range(natm):
        log.info('Follow AOM for atom %d', i)
        dump_tri(sys.stdout, aom[i], ncol=NCOL, digits=DIGITS, start=0)
        totaom += aom[i]
        for j in range(NPROPS):
            log.info('Nuclei %d prop %s value : %8.5f', i, PROPS[j], props[i,j])
            totprops[j] += props[i,j]
    for j in range(NPROPS):
        log.info('Tot prop %s value : %8.5f', PROPS[j], totprops[j])
    log.info('Follow total AOM')
    dump_tri(sys.stdout, totaom, ncol=NCOL, digits=DIGITS, start=0)
    i = numpy.identity(nmo)
    diff = numpy.linalg.norm(totaom-i)
    log.info('Diff in S matrix : %8.5f', diff)

def print_basin(name, natm):                
    props = numpy.zeros((natm,NPROPS))
    totprops = numpy.zeros((NPROPS))
    with h5py.File(name) as f:
        for i in range(natm):
            idx = 'atom_props'+str(i)
            props[i] = f[idx+'/totprops'].value
    for i in range(natm):
        for j in range(NPROPS):
            log.info('Nuclei %d prop %s value : %8.5f', i, PROPS[j], props[i,j])
            totprops[j] += props[i,j]
    for j in range(NPROPS):
        log.info('Tot prop %s value : %8.5f', PROPS[j], totprops[j])

if __name__ == '__main__':
    name = 'lih.chk.h5'
    natm = 2
    for i in range(natm):
        print_surface_txt(name,i)
