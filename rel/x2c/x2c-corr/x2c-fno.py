#!/usr/bin/env python

import numpy, fno
from pyscf import gto, scf, x2c, lib, mp

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.kernel()

ncore = 2
nocc = mol.nelectron - ncore
cv, ev = fno.getfno(mf,ncore,thresh_vir=1e-4)
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = cv
ec = mf.mo_energy[:ncore]
eo = mf.mo_energy[ncore:ncore+nocc]
coeff = numpy.hstack([mo_core,mo_occ,mo_vir])
energy = numpy.hstack([ec,eo,ev])
occ = numpy.zeros(coeff.shape[1])
for i in range(mol.nelectron):
    occ[i] = 1.0

pt = mp.X2CMP2(mf, mo_coeff=coeff, mo_occ=occ)
pt.frozen = ncore
pt.kernel(mo_energy=energy)

