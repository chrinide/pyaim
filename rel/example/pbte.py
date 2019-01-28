#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'pbte'

mol = gto.Mole()
mol.basis = 'unc-ano'
mol.atom = '''
Pb 0.0 0.0 0.00
Te 0.0 0.0 2.5949
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.RDHF(mol)
mf.chkfile = name+'.chk'
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.kernel()

