#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'ceo.chk'
atm = [0,1]

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1, aom2.conj())
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

natocc, natorb = numpy.linalg.eigh(aom1)
log.info('Occ for DAFH %s', natocc)
log.info('Sum Occ for DAFH %s', natocc.sum())

