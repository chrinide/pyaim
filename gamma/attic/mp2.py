#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf
from pyscf import mp, ao2mo

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

pt = mp.RMP2(mf).run()
print("RMP2 energy (per unit cell) at k-point =", pt.e_tot)
rdm1 = pt.make_rdm1()
rdm2 = pt.make_rdm2()
nmo = mf.mo_coeff.shape[1]
eri_mo = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
h1 = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
e_tot = numpy.einsum('ij,ji', h1, rdm1) + numpy.einsum('ijkl,jilk', eri_mo, rdm2)*.5 + mf.energy_nuc()
print("RMP2 energy based on MP2 density matrices =", e_tot.real)

