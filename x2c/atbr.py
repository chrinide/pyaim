#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'atbr'

mol = gto.Mole()
mol.basis = {'At':'unc-ano', 'Br':'unc-ano'}
mol.atom = '''
At 0.0 0.0  0.000
Br 0.0 0.0  2.746
'''
mol.charge = 0
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

