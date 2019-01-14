#!/usr/bin/env python

import numpy
from pyscf import gto, scf, x2c, lib, dft

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
H 0.0 0.0 0.0
H 0.0 0.0 2.0
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
#dm = mf.get_init_guess() + 0.1j
dm = mf.get_init_guess() 
mf.kernel(dm)

print mf.mo_coeff.shape
print mf.mo_coeff.dtype
print mf.mo_occ

dm = mf.make_rdm1()
print dm.dtype, dm.shape
grids = dft.gen_grid.Grids(mol)
grids.kernel()
ao = dft.r_numint.eval_ao(mol, grids.coords, deriv=0, with_s=False)
rho = dft.r_numint.eval_rho(mol, ao, dm, xctype='LDA')
print rho[1].shape
rhoa = numpy.einsum('i,i->',rho[0],grids.weights)
#rhob = numpy.einsum('i,i->',rho[1],grids.weights)
#print rhoa+rhob
print rhoa
