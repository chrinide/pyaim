#!/usr/bin/env python

import numpy
from pyscf import gto, scf, lib, ao2mo

mol = gto.Mole()
mol.basis = '3-21g'
mol.atom = '''
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
    '''
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
ehf = mf.kernel()
nao = mol.nao_nr()
dm = mf.make_rdm1()

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')

enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
lib.logger.info(mf,'* Info on AO basis')
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

nmo = mf.mo_coeff.shape[1]
coeff = mf.mo_coeff[:,:nmo]
rdm1 = numpy.zeros((nmo,nmo))
for i in range(mol.nelectron//2):
    rdm1[i,i] = 2.0

s = coeff.T.dot(s).dot(coeff)
t = coeff.T.dot(t).dot(coeff)
v = coeff.T.dot(v).dot(coeff)
ekin = numpy.einsum('ij,ji->',t,rdm1)
pop = numpy.einsum('ij,ji->',s,rdm1)
elnuce = numpy.einsum('ij,ji->',v,rdm1)
lib.logger.info(mf,'* Info on MO basis')
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

coeff = numpy.linalg.inv(coeff)
s = coeff.T.dot(s).dot(coeff)
t = coeff.T.dot(t).dot(coeff)
v = coeff.T.dot(v).dot(coeff)
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
lib.logger.info(mf,'* Backtransform from MO to AO basis')
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

