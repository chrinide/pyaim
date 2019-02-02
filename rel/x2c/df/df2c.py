#!/usr/bin/env python

import time, numpy
from pyscf import lib, gto, scf, x2c
from pyscf.df import r_incore

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

t = time.time()
cderi = r_incore.cholesky_eri(mol, int3c='int3c2e_spinor', auxbasis='def2-svp-jkfit', verbose=4)
def fjk2c(mol, dm, *args, **kwargs):
    n2c = dm.shape[0]
    cderi_ll = cderi.reshape(-1,n2c,n2c)
    vj = numpy.zeros((n2c,n2c), dtype=dm.dtype)
    vk = numpy.zeros((n2c,n2c), dtype=dm.dtype)
    rho = numpy.dot(cderi, dm.T.reshape(-1))
    vj = numpy.dot(rho, cderi).reshape(n2c,n2c)
    v1 = lib.einsum('pij,jk->pik', cderi_ll, dm)
    vk = lib.einsum('pik,pkj->ij', v1, cderi_ll)
    return vj, vk

mf = x2c.RHF(mol)
mf.get_jk = fjk2c
mf.direct_scf = False
mf.scf()
print('Time %.3f (sec)' % (time.time()-t))

t = time.time()
mf = x2c.RHF(mol).density_fit()
mf.with_df.auxbasis = 'def2-svp-jkfit'
mf.verbose = 3
mf.scf()
print('Time %.3f (sec)' % (time.time()-t))

