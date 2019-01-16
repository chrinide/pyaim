#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf
from pyscf import cc, mp, ao2mo

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'sto-3g'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.build()

mf = scf.RHF(cell).density_fit(auxbasis='def2-svp-jkfit')
mf.exxdiv = None
ehf = mf.kernel()

mcc = cc.CCSD(mf)
mcc.frozen = 2
mcc.kernel()

