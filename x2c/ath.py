#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'ath'

mol = gto.Mole()
mol.basis = {'At':'unc-ano','H':'unc-tzp-dk'}
mol.atom = '''
At 0.0 0.0  0.000
H  0.0 0.0  1.750
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

