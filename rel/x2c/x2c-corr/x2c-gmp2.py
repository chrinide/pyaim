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
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
dm = mf.get_init_guess() + 0.1j
ex2c = mf.kernel()

ncore = 2
pt = mp.X2CMP2(mf)
pt.frozen = ncore
pt.kernel()
#rdm1 = pt.make_rdm1()
#rdm2 = pt.make_rdm2()
#c = mf.mo_coeff
#nmo = mf.mo_coeff.shape[1]
#eri_ao = mol.intor('int2e_spinor')
#eri_mo = ao2mo.general(eri_ao,(c,c,c,c)).reshape(nmo,nmo,nmo,nmo)
#hcore = mf.get_hcore()
#h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
#e = numpy.einsum('ij,ji', h1, rdm1)
#e += numpy.einsum('ijkl,ijkl', eri_mo, rdm2)*0.5
#e += mol.energy_nuc()
#lib.logger.info(mf,"!*** E(MP2) with RDM: %s" % e)

#rdm1 = pt.make_rdm1_vv()
#natocc, natorb = numpy.linalg.eigh(-rdm1)
#natocc = -natocc
#lib.logger.info(mf,"* Occupancies")
#lib.logger.info(mf,"* %s" % natocc)
#lib.logger.info(mf,"* The sum is %8.6f" % numpy.sum(natocc)) 

