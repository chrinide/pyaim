#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib

name = 'auh'

mol = gto.Mole()
mol.basis = {'Au':'unc-ano','H':'unc-tzp-dk'}
mol.atom = '''
Au 0.0 0.0  0.000
H  0.0 0.0  1.524
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.RHF(mol).x2c()
mf.chkfile = name+'.chk'
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

