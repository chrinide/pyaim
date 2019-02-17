#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf
from pyscf import fci, ao2mo, mcscf

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'def2-svp'
cell.precision = 1e-6
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

from pyscf.mcscf import avas
ao_labels = ['C 2s', 'C 2p']
norb, ne_act, orbs = avas.avas(mf, ao_labels, canonicalize=True, ncore=2)

mc = mcscf.CASSCF(mf, norb, ne_act)
#mc.fcisolver = fci.selected_ci_spin0.SCI()
#mc.fix_spin_(shift=.5, ss=0.0000)
#mc.fcisolver.ci_coeff_cutoff = 0.005
#mc.fcisolver.select_cutoff = 0.005
mc.kernel(orbs)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

#natocc, natorb = numpy.linalg.eigh(-rdm1)
#for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
#    if natorb[k,i] < 0:
#        natorb[:,i] *= -1
#natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
#natocc = -natocc
#
#wfn_file = name + '.wfn'
#with open(wfn_file, 'w') as f2:
#    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
#    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
#    wfn_format.write_ci(f2, select_ci.to_fci(mc.ci,mc.ncas,mc.nelecas), mc.ncas, mc.nelecas, ncore=mc.ncore)

