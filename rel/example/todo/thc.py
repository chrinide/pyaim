#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft

name = 'thc'

mol = gto.Mole()
mol.basis = {'Th':'unc-ano','C':'unc-tzp-dk'}
mol.atom = '''
Th 0.0 0.0  0.000
C  0.0 0.0  1.967
'''
mol.charge = 0
mol.spin = 2
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = scf.DHF(mol)
mf.chkfile = name+'.chk'
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
#mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
#dm = mf.make_rdm1()
mf.kernel()

