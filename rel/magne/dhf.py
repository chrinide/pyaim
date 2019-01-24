#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'dhf'

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
Po     0.000000      0.000000      0.418351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.DHF(mol)
dm = mf.get_init_guess() + 0.1j
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.chkfile = name+'.chk'
mf.kernel(dm)

dm = mf.make_rdm1()
nao = mf.mo_occ.shape
n2c = mol.nao_2c()
c1 = 0.5/lib.param.LIGHT_SPEED
dmLL = dm[:n2c,:n2c].copy('C')
dmSS = dm[n2c:,n2c:] * c1**2

#rho += rhoS
#M = |\beta\Sigma|
#m[0] -= mS[0]
#m[1] -= mS[1]
#m[2] -= mS[2]
#s = lib.norm(m, axis=0)
#rhou = (r + s) * .5
#rhod = (r - s) * .5
#rho = (rhou, rhod)

coords = numpy.zeros(3)
coords = coords.reshape(-1,3)
aoLS = dft.r_numint.eval_ao(mol, coords, deriv=0)
rho = dft.r_numint.eval_rho(mol, aoLS[:2], dmLL, xctype='LDA')
rhoS = dft.r_numint.eval_rho(mol, aoLS[2:], dmSS, xctype='LDA')
print rho
print rhoS
print rho+rhoS

