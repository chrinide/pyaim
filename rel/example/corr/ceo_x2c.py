#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, x2c

name = 'ceo_x2c'

mol = gto.Mole()
mol.basis = 'x2c-tzvpp'
mol.atom = '''
Ce 0.0 0.0  0.0 
O  0.0 0.0  1.820369737
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.nucmod = 0
mol.build()

mf = x2c.RHF(mol).density_fit()
mf.chkfile = name+'.chk'
mf.with_df.auxbasis = 'x2c-jfit'
mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
dm = mf.make_rdm1(dm)
mf.kernel()

ncore = 52

pt = x2c.MP2(mf)
pt.frozen = ncore
pt.kernel()
rdm1 = pt.make_rdm1()
rdm2 = pt.make_rdm2()

#cc = x2c.CCSD(mf)
#cc.frozen = ncore
#cc.kernel()

lib.logger.info(mf,'Write rdms on MO basis to HDF5 file')
dic = {'rdm1':rdm1, 'rdm2':rdm2}
lib.chkfile.save(name+'.chk', 'rdm', dic)

