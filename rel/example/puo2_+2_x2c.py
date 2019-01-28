#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'puo2_+2_x2c'

mol = gto.Mole()
mol.basis = {'Pu':'unc-ano','O':'unc-tzp-dk'}
mol.atom = '''
Pu 0.0 0.0  0.0 
O  0.0 0.0  1.645
O  0.0 0.0 -1.645
'''
mol.charge = 2
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.UHF(mol)
mf.chkfile = name+'.chk'
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

