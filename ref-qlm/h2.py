#!/usr/bin/env python

from pyscf import gto, scf
from pyscf.tools import wfn_format

name = 'h2'

mol = gto.Mole()
mol.basis = 'sto-3g'
mol.atom = '''
H      0.000000      0.000000      0.000000
H      0.000000      0.000000      0.750000
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], \
    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])

