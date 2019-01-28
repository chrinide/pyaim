#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'pbo'

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

mf = scf.RHF(mol).x2c()
mf.chkfile = name+'.chk'
mf.kernel()

