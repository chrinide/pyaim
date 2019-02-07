#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'test.chk'
atm = [0,1]

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

with h5py.File(name) as f:
    rdm1 = f['rdm/rdm1'].value
    rdm2 = f['rdm/rdm2'].value

print rdm2.shape
rdm2 = rdm2 - lib.einsum('ij,kl->ijkl',rdm1,rdm1) 
rdm2 = -rdm2

delta = numpy.einsum('ijkl,ji,lk->', rdm2, aom1, aom2.conj(), optimize=True)
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

