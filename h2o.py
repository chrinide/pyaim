#!/usr/bin/env python

import numpy, time
from pyscf import gto, scf, lib, dft

name = 'h2o'

mol = gto.Mole()
mol.atom = '''
      o     0    0       0
      h     0    -.757   .587
      h     0    .757    .587
'''
mol.basis = 'aug-cc-pvqz'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = dft.RKS(mol)
mf.max_cycle = 150
mf.grids.atom_grid = {'H': (20,110), 'O': (20,110)}
mf.grids.prune = None
mf.chkfile = name+'.chk'
mf.xc = 'rpw86,pbe'
mf.kernel()

dm = mf.make_rdm1()
coords = mf.grids.coords
weights = mf.grids.weights
ngrids = len(weights)
ao = dft.numint.eval_ao(mol, coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
lib.logger.info(mf,'Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))
ex, vx = dft.libxc.eval_xc('rPW86,', rho)[:2]
ec, vc = dft.libxc.eval_xc(',PBE', rho)[:2]
lib.logger.info(mf, 'Ex  = %.12f' % numpy.einsum('i,i,i->', ex, rho[0], weights))
lib.logger.info(mf, 'Ec  = %.12f' % numpy.einsum('i,i,i->', ec, rho[0], weights))
lib.logger.info(mf, 'Exc = %.12f' % numpy.einsum('i,i,i->', ex+ec, rho[0], weights))

