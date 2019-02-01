#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c, dft

name = 'pbo_x2c_lda'

mol = gto.Mole()
mol.basis = {'Pb':'unc-ano','O':'unc-tzp-dk'}
mol.atom = '''
Pb 0.0 0.0 0.00
O  0.0 0.0 1.922
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.RKS(mol)
mf.chkfile = name+'.chk'
mf.grids.level = 5
mf.grids.prune = None
mf.kernel()

