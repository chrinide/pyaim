#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'dhf'

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
mol.nucmod = 1
mol.build()

coords = numpy.zeros(3)
coords = coords.reshape(-1,3)
mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
dm = mf.make_rdm1()
ao = dft.numint.eval_ao(mol, coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm, xctype='GGA')
print "Ref non-rel value", rho

