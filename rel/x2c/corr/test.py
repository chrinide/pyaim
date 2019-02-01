#!/usr/bin/env python

from pyscf import scf, x2c
from pyscf import gto
import x2cmp2

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
Pb 0.0 0.0 0.00
O  0.0 0.0 1.922
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = x2c.RHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.kernel()

ncore = 110

pt = x2cmp2.GMP2(mf)
pt.frozen = ncore
pt.kernel()

mo_coeff, mo_energy, mo_occ = pt.fno()

pt = x2cmp2.GMP2(mf, mo_coeff=mo_coeff, mo_occ=mo_occ)
pt.frozen = ncore
pt.kernel(mo_energy=mo_energy)

