#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, dft
from pyscf.tools import wfn_format

name = 'prueba'

mol = gto.Mole()
mol.basis = 'unc-dzp'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

wfn_file = name+'.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], \
    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])
