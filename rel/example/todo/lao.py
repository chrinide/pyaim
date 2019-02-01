#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'lao'

mol = gto.Mole()
mol.basis = {'La':'unc-ano','O':'unc-tzp-dk'}
mol.atom = '''
La 0.0 0.0 0.00
O  0.0 0.0 1.825661509 
'''
mol.charge = 0
mol.spin = 1
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.DHF(mol)
mf.chkfile = name+'.chk'
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.kernel()

