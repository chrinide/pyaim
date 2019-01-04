#!/usr/bin/env python

from pyscf import gto, scf, lib, dft, ao2mo

name = 'h2o'

mol = gto.Mole()
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.kernel()

s = mol.intor('int1e_ovlp')
lib.logger.info(mf,'Write aom on AO basis to HDF5 file')
atom_dic = {'overlap':s}
lib.chkfile.save(name+'_integrals.h5', 'molecule', atom_dic)

