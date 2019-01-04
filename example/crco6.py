#!/usr/bin/env python

import numpy, time, h5py, os, sys
from pyscf import gto, scf, lib, dft, ao2mo

name = 'crco6'
bco = 1.14
bcc = 2.0105

mol = gto.Mole()
mol.atom = [
    ['Cr',(  0.000000,  0.000000,  0.000000)],
    ['C', (  bcc     ,  0.000000,  0.000000)],
    ['O', (  bcc+bco ,  0.000000,  0.000000)],
    ['C', ( -bcc     ,  0.000000,  0.000000)],
    ['O', ( -bcc-bco ,  0.000000,  0.000000)],
    ['C', (  0.000000,  bcc     ,  0.000000)],
    ['O', (  0.000000,  bcc+bco ,  0.000000)],
    ['C', (  0.000000, -bcc     ,  0.000000)],
    ['O', (  0.000000, -bcc-bco ,  0.000000)],
    ['C', (  0.000000,  0.000000,  bcc     )],
    ['O', (  0.000000,  0.000000,  bcc+bco )],
    ['C', (  0.000000,  0.000000, -bcc     )],
    ['O', (  0.000000,  0.000000, -bcc-bco )],
]
dirnow = os.path.realpath(os.path.join(__file__, '..'))
basfile = os.path.join(dirnow, 'sqzp.dat')
mol.basis = basfile
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol).density_fit() 
#mf.with_df.auxbasis = 'aug-cc-pvdz-jkfit'
mf.max_cycle = 150
mf.chkfile = name+'.chk'
ehf = mf.kernel()

