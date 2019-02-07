#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'uo2_+2_x2c'

mol = gto.Mole()
mol.basis = {'U':'ano','O':'tzp-dk'}
mol.atom = '''
U 0.0 0.0  0.0 
O  0.0 0.0  1.708
O  0.0 0.0 -1.708
'''
mol.charge = 2
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.RHF(mol).density_fit()
mf.with_x2c.basis = {'U':'dyalltz','O':'unc-ano'}
mf.chkfile = name+'.chk'
mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
dm = mf.make_rdm1()
mf.kernel(dm)

ncore = 86

pt = x2c.MP2(mf)
pt.frozen = ncore
pt.kernel()

mo_coeff, mo_energy, mo_occ = pt.fno()

pt = x2c.MP2(mf, mo_coeff=mo_coeff, mo_occ=mo_occ)
pt.frozen = ncore
pt.kernel(mo_energy=mo_energy)
#rdm1 = pt.make_rdm1()
#rdm2 = pt.make_rdm2()

#cc = x2c.CCSD(mf)
#cc.frozen = ncore
#cc.kernel()

#lib.logger.info(mf,'Write rdms on MO basis to HDF5 file')
#dic = {'rdm1':rdm1, 'rdm2':rdm2}
#lib.chkfile.save(name+'.chk', 'rdm', dic)

