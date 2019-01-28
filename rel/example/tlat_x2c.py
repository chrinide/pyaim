#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'tlat_x2c'

mol = gto.Mole()
mol.basis = 'unc-ano'
mol.atom = '''
Tl 0.0 0.0 0.0000
At 0.0 0.0 2.9773
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.UHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

