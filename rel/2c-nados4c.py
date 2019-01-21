#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'dhf.chk'
atm = [0,1]
nmo = 20

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1.conj(), aom2)
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

mol = lib.chkfile.load_mol(name)

dab = 2*numpy.einsum('ik,kj->ij', aom1, aom2)
dba = 2*numpy.einsum('ik,kj->ij', aom2, aom1)
d2c = (dab+dba)/2.0

natocc, natorb = numpy.linalg.eigh(d2c)
log.info('Occ for NADO %s', natocc)
log.info('Sum Occ for NADO %f', natocc.sum())

