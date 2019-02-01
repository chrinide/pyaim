#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'pbo_x2c.chk'
atm = [0,1]

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1, aom2.conj())
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

dab = 2*numpy.einsum('ik,kj->ij', aom1, aom2.conj())
dba = 2*numpy.einsum('ik,kj->ij', aom2, aom1.conj())
d2c = (dab+dba)/2.0

natocc, natorb = numpy.linalg.eigh(d2c)
log.info('Occ for NADO %s', natocc)
log.info('Sum Occ for NADO %f', natocc.sum())
