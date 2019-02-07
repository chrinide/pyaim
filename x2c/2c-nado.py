#!/usr/bin/env python

import h5py
import numpy
import sys

if sys.version_info >= (3,):
    unicode = str

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'pbpo'
atm = [0,1]
mol = lib.chkfile.load_mol(name+'.chk')
mo_coeff = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
nmo = mol.nelectron//2
mo_coeff = mo_coeff[:,0:nmo]

with h5py.File(name+'.chk.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 4*numpy.einsum('ij,ji->', aom1, aom2)
log.info('Delta %f for pair %d %d' %  (delta, atm[0], atm[1]))

dab = 4*numpy.einsum('ik,kj->ij', aom1, aom2)
dba = 4*numpy.einsum('ik,kj->ij', aom2, aom1)
d2c = (dab+dba)/2.0

from pyscf.tools import molden
natocc, natorb = numpy.linalg.eigh(d2c)
log.info('Occ for NADO %s', natocc)
log.info('Sum Occ for NADO %f', natocc.sum())
natorb = numpy.dot(mo_coeff, natorb)
with open(name+'_'+str(atm[0])+'-'+str(atm[1])+'_2c.molden', 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, natorb, occ=natocc)

