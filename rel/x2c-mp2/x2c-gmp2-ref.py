#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, x2c
from pyscf import ao2mo, lib, mp

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
ex2c = mf.kernel()

eri_ao = mol.intor('int2e_spinor')
oidx = mf.mo_occ>0
vidx = mf.mo_occ<=0
nao = eri_ao.shape[0]
nocc = mol.nelectron
nvir = nao - nocc
nmo = nocc + nvir
co = mf.mo_coeff[:,oidx]
cv = mf.mo_coeff[:,vidx]
c = numpy.hstack((co,cv))
o = slice(0, nocc)
v = slice(nocc, None)
eo = mf.mo_energy[:nocc]
ev = mf.mo_energy[nocc:]

pt = mp.X2CMP2(mf)
pt.kernel()
rdm1 = pt.make_rdm1()
rdm2 = pt.make_rdm2()
eri_mo = ao2mo.general(eri_ao,(c,c,c,c)).reshape(nmo,nmo,nmo,nmo)
hcore = mf.get_hcore()
h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
e = numpy.einsum('ij,ji', h1, rdm1)
e += numpy.einsum('ijkl,ijkl', eri_mo, rdm2)*0.5
e += mol.energy_nuc()
lib.logger.info(mf,"!*** E(MP2) with RDM: %s" % e)

lib.logger.info(mf,'**** Relativistic GMP2 Ref values')
eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
e_denom = 1.0/(-ev.reshape(-1,1,1,1)+eo.reshape(-1,1,1)-ev.reshape(-1,1)+eo)
t2 = numpy.zeros((nvir,nocc,nvir,nocc))
t2 = eri_mo[v,o,v,o]*e_denom 
e_mp2 = 0.25*numpy.einsum('iajb,aibj->', eri_mo[o,v,o,v], t2, optimize=True)
lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mf,"!*** E(X2C+MP2): %s" % (e_mp2+ex2c))

