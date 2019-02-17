#!/usr/bin/env python

import h5py
import numpy
import sys

if sys.version_info >= (3,):
    unicode = str

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'kpts_hf'
atm = [0,1]

with h5py.File(name+'.chk.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 4*numpy.einsum('ij,ji->', aom1, aom2)
log.info('Delta %f for pair %d %d' %  (delta, atm[0], atm[1]))

log.info('DAFH for atom %d', atm[0])
natocc, natorb = numpy.linalg.eigh(aom1)
natocc *= 2.0
log.info('Occ for DAFH %s', natocc)
log.info('Sum Occ for DAFH %f', natocc.sum())

