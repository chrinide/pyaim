#!/usr/bin/env python

import time, numpy
from pyscf import lib, gto, scf, x2c
from pyscf.df.r_incore import cholesky_eri

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

cderi = (cholesky_eri(mol, int3c='int3c2e_spinor', auxbasis='def2-svp-jkfit', verbose=4),
         cholesky_eri(mol, int3c='int3c2e_spsp1_spinor', auxbasis='def2-svp-jkfit', verbose=4))
def fjk4c(mol, dm, *args, **kwargs):
    n2c = dm.shape[0]//2
    c2 = 0.5 / lib.param.LIGHT_SPEED
    cderi_ll = cderi[0].reshape(-1,n2c,n2c)
    cderi_ss = cderi[1].reshape(-1,n2c,n2c)
    vj = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)
    vk = numpy.zeros((n2c*2,n2c*2), dtype=dm.dtype)
    rho = (numpy.dot(cderi[0], dm[:n2c,:n2c].T.reshape(-1))
         + numpy.dot(cderi[1], dm[n2c:,n2c:].T.reshape(-1)*c2**2))
    vj[:n2c,:n2c] = numpy.dot(rho, cderi[0]).reshape(n2c,n2c)
    vj[n2c:,n2c:] = numpy.dot(rho, cderi[1]).reshape(n2c,n2c) * c2**2
    v1 = lib.einsum('pij,jk->pik', cderi_ll, dm[:n2c,:n2c])
    vk[:n2c,:n2c] = lib.einsum('pik,pkj->ij', v1, cderi_ll)
    v1 = lib.einsum('pij,jk->pik', cderi_ss, dm[n2c:,n2c:])
    vk[n2c:,n2c:] = lib.einsum('pik,pkj->ij', v1, cderi_ss) * c2**4
    v1 = lib.einsum('pij,jk->pik', cderi_ll, dm[:n2c,n2c:])
    vk[:n2c,n2c:] = lib.einsum('pik,pkj->ij', v1, cderi_ss) * c2**2
    vk[n2c:,:n2c] = vk[:n2c,n2c:].T.conj()
    return vj, vk

t = time.time()
mf = scf.RDHF(mol)
mf.get_jk = fjk4c
mf.direct_scf = False
mf.scf()
print('Time %.3f (sec)' % (time.time()-t))

t = time.time()
mf = scf.RDHF(mol).density_fit()
mf.verbose = 3
mf.scf()
print('Time %.3f (sec)' % (time.time()-t))

t = time.time()
mf = scf.RDHF(mol)
mf.verbose = 3
mf.scf()
print('Time %.3f (sec)' % (time.time()-t))
