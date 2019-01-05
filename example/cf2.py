#!/usr/bin/env python

import numpy, sys
from pyscf import gto, scf, cc, lib

name = 'cf2_ccsd'

mol = gto.Mole()
mol.basis = 'aug-cc-pvdz'
mol.atom = '''
C      0.000000      0.000000      0.262523
F      0.000000      1.032606     -0.541812
F      0.000000     -1.032606     -0.541812
'''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

ncore = 3
mcc = cc.CCSD(mf)
mcc.direct = 1
mcc.diis_space = 10
mcc.frozen = ncore
mcc.conv_tol = 1e-6
mcc.conv_tol_normt = 1e-6
mcc.max_cycle = 150
mcc.kernel()

t1norm = numpy.linalg.norm(mcc.t1)
t1norm = t1norm/numpy.sqrt(mol.nelectron-ncore*2)
lib.logger.info(mcc,"* T1 norm should be les than 0.02")
lib.logger.info(mcc,"* T1 norm : %12.6f" % t1norm)

rdm1 = mcc.make_rdm1()
lib.logger.info(mf,'Write rdm1 on MO basis to HDF5 file')
dic = {'rdm1':rdm1}
lib.chkfile.save(name+'.chk', 'rdm', dic)

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
s = mf.mo_coeff.T.dot(s).dot(mf.mo_coeff)
t = mf.mo_coeff.T.dot(t).dot(mf.mo_coeff)

ekin = numpy.einsum('ij,ji->',t,rdm1)
pop = numpy.einsum('ij,ji->',s,rdm1)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)

