#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'prueba.chk'
atm = [0,1]
nmo = 5
aom1 = numpy.zeros((nmo,nmo))
aom2 = numpy.zeros((nmo,nmo))

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 4*numpy.einsum('ij,ji->', aom1, aom2)
log.info('Delta %f for pair %d %d' %  (delta, atm[0], atm[1]))

#TODO:descompose delta in the basis of dafh which is diagonal

mol = lib.chkfile.load_mol(name)
mo_coeff = lib.chkfile.load(name, 'scf/mo_coeff')
mo_coeff = mo_coeff[:,0:nmo]

from pyscf.tools import molden
from pyscf import symm
#orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff)
#natocc, natorb = symm.eigh(aom1, orbsym)
natocc, natorb = numpy.linalg.eigh(aom1)
natocc *= 2.0
log.info('Occ for DAFH %s', natocc)
log.info('Sum Occ for DAFH %f', natocc.sum())
natorb = numpy.dot(mo_coeff, natorb)
#natorb_sym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, natorb)
#print(natorb_sym)

with open('h2o.molden', 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, natorb, occ=natocc)

