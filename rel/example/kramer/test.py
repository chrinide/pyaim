#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

mol = gto.Mole()
mol.basis = 'dzp-dk'
mol.atom = '''
Pb 0.0 0.0 0.00
O  0.0 0.0 1.922
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.RHF(mol)
mf.kernel()

# T = -i \sigma_y K_0
# K_0 -> complex conjugate
#rab = numpy.einsum('i,i->p', c0b, c0a.conj()) 
#my = rba.imag - rab.imag
