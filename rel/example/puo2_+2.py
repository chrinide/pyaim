#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'puo2_+2'

mol = gto.Mole()
mol.basis = {'Pu':'unc-ano', 'O':'unc-tzp-dkh'}
mol.atom = '''
Pu 0.0 0.0 0.0 
O  0.0 0.0 1.645
O  0.0 0.0 -1.645
'''
mol.charge = 2
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.DHF(mol).apply(scf.addons.remove_linear_dep_) 
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.chkfile = name+'.chk'
mf.kernel()

grids = dft.gen_grid.Grids(mol)
grids.kernel()
dm = mf.make_rdm1()
coords = grids.coords
weights = grids.weights

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

aoLS = eval_ao(mol, coords, deriv=1)
rho = eval_rho(mol, aoLS[:2], dmLL, xctype='GGA')
rhoS = eval_rho(mol, aoLS[2:], dmSS, xctype='GGA')
print('RhoL = %.12f' % numpy.einsum('i,i->', rho[0], weights))
print('RhoS = %.12f' % numpy.einsum('i,i->', rhoS[0], weights))
print('Rho = %.12f' % numpy.einsum('i,i->', rho[0]+rhoS[0], weights))

coords = numpy.zeros(3)
coords = coords.reshape(-1,3)
aoLS = eval_ao(mol, coords, deriv=1)
rho = eval_rho(mol, aoLS[:2], dmLL, xctype='GGA')
rhoS = eval_rho(mol, aoLS[2:], dmSS, xctype='GGA')
print rho
print rhoS
print rho+rhoS
rho2 = eval_rho2(mol, aoLS[:2], mf.mo_coeff, mf.mo_occ, small=False, xctype='GGA')
rho2S = eval_rho2(mol, aoLS[2:], mf.mo_coeff, mf.mo_occ, small=True, xctype='GGA')
print rho2
print rho2S
print rho2+rho2S

