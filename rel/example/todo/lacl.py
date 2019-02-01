#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'lacl'

mol = gto.Mole()
mol.basis = {'La':'unc-ano','Cl':'unc-tzp-dk'}
mol.atom = '''
La 0.0 0.0 0.000
Cl 0.0 0.0 2.4981
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

