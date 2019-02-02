#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'at2_x2c'

mol = gto.Mole()
mol.basis = {'At':'unc-ano'}
mol.atom = '''
At 0.0 0.0  0.000
At 0.0 0.0  3.100
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.RHF(mol)
mf.chkfile = name+'.chk'
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

