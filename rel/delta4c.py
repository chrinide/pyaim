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
aom1 = numpy.zeros((nmo,nmo))
aom2 = numpy.zeros((nmo,nmo))

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1.conj(), aom2)
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

mol = lib.chkfile.load_mol(name)
#mol.symmetry = 0
#mo_coeff = lib.chkfile.load(name, 'scf/mo_coeff')
#mo_coeff = mo_coeff[:,0:10]
natocc, natorb = numpy.linalg.eigh(aom1)
#natorb = numpy.dot(mo_coeff, natorb)
log.info('Occ for DAFH %s', natocc)
log.info('Sum Occ for DAFH %s', natocc.sum())

#from pyscf.tools import molden
#with open('x2c.molden', 'w') as f1:
#    molden.header(mol, f1)
#    molden.orbital_coeff(mol, f1, natorb, occ=natocc)

