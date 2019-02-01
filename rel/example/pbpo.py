#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'pbpo'

mol = gto.Mole()
mol.basis = 'unc-ano'
mol.atom = '''
Pb 0.0 0.0  0.000
Po 0.0 0.0  3.295
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
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

