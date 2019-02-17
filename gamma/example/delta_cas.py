#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'gamma_cas.chk'
atm = [0,1]

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

with h5py.File(name) as f:
    rdm1 = f['pdm/rdm1'].value
    rdm2 = f['pdm/rdm2'].value

rdm2 = rdm2 - lib.einsum('ij,kl->ijkl',rdm1,rdm1) 
rdm2 = -rdm2

delta = 2.0*numpy.einsum('ijkl,ji,lk->', rdm2, aom1, aom2, optimize=True)
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

