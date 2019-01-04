#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, dft
from pyscf.tools import wfn_format

name = 'c2f4'

mol = gto.Mole()
mol.basis = 'aug-cc-pvtz'
mol.atom = open('../geom/c2f4.xyz').read()
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = dft.RKS(mol).apply(scf.addons.remove_linear_dep_)
mf.direct_scf = True
mf.conv_tol = 1e-8
mf.grids.radi_method = dft.mura_knowles
mf.grids.becke_scheme = dft.stratmann
mf.grids.level = 4
mf.grids.prune = None
mf.xc = 'm06l,m06l'
mf.kernel()

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], mf.mo_occ[mf.mo_occ>0])
