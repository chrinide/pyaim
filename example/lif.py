#!/usr/bin/env python

import numpy, time, h5py, os, sys
from pyscf import gto, scf, lib, dft, ao2mo
from pyscf.tools import wfn_format

name = 'lif'

mol = gto.Mole()
mol.atom = '''
F  0.0000  0.0000  0.0000
Li 0.0000  0.0000  1.5639
'''
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.kernel()

