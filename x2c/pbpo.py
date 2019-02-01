#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'pbpo'

mol = gto.Mole()
mol.basis = 'unc-ano'
mol.atom = '''
Pb 0.0 0.0 0.00
Po 0.0 0.0 3.295
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.RHF(mol).x2c()
mf.chkfile = name+'.chk'
mf.kernel()

