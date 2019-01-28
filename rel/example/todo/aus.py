#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'auo'

mol = gto.Mole()
mol.basis = {'Au':'unc-ano','O':'unc-tzp-dk'}
mol.atom = '''
Au 0.0 0.0 0.00
S  0.0 0.0 2.156
'''
mol.charge = 0
mol.spin = 1
mol.symmetry = 1
mol.verbose = 4
mol.nucmod = 0
mol.build()

#mf = scf.GHF(mol).x2c()
#mf.chkfile = name+'_x2c_scalar.chk'
#mf.__dict__.update(scf.chkfile.load(name+'_x2c.chk', 'scf'))
#dm = mf.make_rdm1()
#mf.kernel()

mf = x2c.UHF(mol)
mf.chkfile = name+'_x2c.chk'
mf.__dict__.update(scf.chkfile.load(name+'_x2c.chk', 'scf'))
dm = mf.make_rdm1()
mf.kernel(dm)

mf = scf.DHF(mol)
mf.chkfile = name+'_dhf.chk'
mf.__dict__.update(scf.chkfile.load(name+'_dhf.chk', 'scf'))
dm = mf.make_rdm1()
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.kernel(dm)

