#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'prueba'

mol = gto.Mole()
mol.basis = 'unc-dzp'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

