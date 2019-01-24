#!/usr/bin/env python

from pyscf import gto, scf, dft

name = 'ecp'

mol = gto.Mole()
mol.atom = '''
S      0.000000      0.000000      0.158351
H      0.000000      0.861187     -0.569725
H      0.000000     -0.861187     -0.569725
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.basis= {'S':'lanl2dz', 'H':'cc-pvdz'} 
mol.ecp = {'S':'lanl2dz'} 
mol.charge = 0
mol.build()

mf = dft.RKS(mol)
mf.chkfile = name+'.chk'
mf.max_cycle = 150
mf.xc = 'rpw86,pbe'
mf.grids.level = 4
mf.kernel()

for i in range(mol.natm):
    print mol.atom_nelec_core(i)
