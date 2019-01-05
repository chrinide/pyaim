#!/usr/bin/env python

from pyscf import gto, scf

name = 'h2'

mol = gto.Mole()
mol.basis = 'sto-3g'
mol.atom = '''
H      0.000000      0.000000      0.000000
H      0.000000      0.000000      0.750000
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

