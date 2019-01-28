#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib

name = 'tlat'

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

mf = scf.RHF(mol).x2c()
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.kernel()

