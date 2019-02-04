#!/usr/bin/env python

from pyscf import gto, scf

name = 'h3rel'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.spin = 1
mol.atom = open('geom/h3_1.0.xyz').read()
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.DHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()
mf.analyze()

