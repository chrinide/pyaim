#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'puo2_+2'

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

mf = scf.RHF(mol).x2c()
mf.chkfile = name+'.chk'
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

