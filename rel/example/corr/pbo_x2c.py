#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'pbo_x2c'

mol = gto.Mole()
mol.basis = 'x2c-tzvpp'
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

mf = x2c.RHF(mol).density_fit()
mf.chkfile = name+'.chk'
mf.with_df.auxbasis = 'x2c-jfit'
mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
dm = mf.make_rdm1()
mf.kernel(dm)

ncore = 52

pt = x2c.MP2(mf)
pt.frozen = ncore
pt.kernel()

cc = x2c.CCSD(mf)
cc.frozen = ncore
cc.kernel()
