#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf

name = 'n2_rhf'

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

