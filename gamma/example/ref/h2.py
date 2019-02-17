#!/usr/bin/env python

import numpy, h5py, avas
from pyscf import gto, scf, dft, lib, mcscf
from pyscf.tools import wfn_format

name = 'h2'
mol = gto.Mole()
mol.atom = '''
H      0.000000      0.000000      0.000000
H      1.000000      0.000000      0.000000
'''
mol.basis = 'def2-svp'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-6
mf.max_cycle = 120
mf.kernel()

aolst1  = ['H 1s']
aolst = aolst1
ncas, nelecas, mo = avas.avas(mf, aolst, threshold_occ=0.1, threshold_vir=0.1, \
                              minao='ano', ncore=0, with_iao=True)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.chkfile = name+'.chk'
mc.max_cycle_macro = 45
mc.max_cycle_micro = 9
emc = mc.kernel(mo)[0]

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

natocc, natorb = numpy.linalg.eigh(-rdm1)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc

wfn_file = name + '_cas.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
    wfn_format.write_ci(f2, mc.ci, mc.ncas, mc.nelecas, ncore=mc.ncore)

