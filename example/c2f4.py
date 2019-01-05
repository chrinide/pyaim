#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, dft

name = 'c2f4'

mol = gto.Mole()
mol.basis = 'def2-qzvppd'
mol.atom = '''
C     -0.662614     -0.000000     -0.000000
C      0.662614     -0.000000     -0.000000
F     -1.388214     -1.100388      0.000000
F      1.388214     -1.100388      0.000000
F     -1.388214      1.100388      0.000000
F      1.388214      1.100388      0.000000
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = dft.RKS(mol)
mf.chkfile = name+'.chk'
mf.grids.level = 4
mf.grids.prune = None
mf.xc = 'pbe0'
mf.kernel()

