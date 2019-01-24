#!/usr/bin/env python

import numpy
from pyscf import gto, scf, x2c, lib, dft

name = 'x2c'

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
Po     0.000000      0.000000      0.418351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.chkfile = name+'.chk'
mf.kernel(dm)

dm = mf.make_rdm1()
a = numpy.zeros(3)
a = a.reshape(-1,3)
ao = dft.r_numint.eval_ao(mol, a, with_s=False, deriv=0)
rho = dft.r_numint.eval_rho(mol, ao, dm)
print rho

