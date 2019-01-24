#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'auh_x2c'

mol = gto.Mole()
mol.basis = {'Au':'dyallqz','H':'unc-tzp-dk'}
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

mf = x2c.UHF(mol)
mf.chkfile = name+'.chk'
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

