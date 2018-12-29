#!/usr/bin/env python

import numpy, time, h5py
from pyscf import gto, scf, lib, dft, ao2mo

name = 'crco6'

mol = gto.Mole()
mol.verbose = 4
bco = 1.14
bcc = 2.0105
mol.atom = [
    ['Cr',(  0.000000,  0.000000,  0.000000)],
    ['C', (  bcc     ,  0.000000,  0.000000)],
    ['O', (  bcc+bco ,  0.000000,  0.000000)],
    ['C', ( -bcc     ,  0.000000,  0.000000)],
    ['O', ( -bcc-bco ,  0.000000,  0.000000)],
    ['C', (  0.000000,  bcc     ,  0.000000)],
    ['O', (  0.000000,  bcc+bco ,  0.000000)],
    ['C', (  0.000000, -bcc     ,  0.000000)],
    ['O', (  0.000000, -bcc-bco ,  0.000000)],
    ['C', (  0.000000,  0.000000,  bcc     )],
    ['O', (  0.000000,  0.000000,  bcc+bco )],
    ['C', (  0.000000,  0.000000, -bcc     )],
    ['O', (  0.000000,  0.000000, -bcc-bco )],
]

mol.basis = {'Cr': 'ccpvtz',
             'C' : 'ccpvtz',
             'O' : 'ccpvtz',}
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.level_shift = .5
mf.diis_start_cycle = 2
mf.conv_tol = 1e-9
mf.chkfile = name+'.chk'
mf.__dict__.update(scf.chkfile.load(name+'.chk', 'scf'))
dm = mf.make_rdm1()
ehf = mf.kernel(dm)

dm = mf.make_rdm1()
nao = mol.nao_nr()

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')

enuc = mol.energy_nuc() 
ekin = numpy.einsum('ij,ji->',t,dm)
pop = numpy.einsum('ij,ji->',s,dm)
elnuce = numpy.einsum('ij,ji->',v,dm)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)
lib.logger.info(mf,'Nuclear Atraction energy : %12.6f' % elnuce)
lib.logger.info(mf,'Nuclear Repulsion energy : %12.6f' % enuc)

lib.logger.info(mf,'Write aom on AO basis to HDF5 file')
atom_dic = {'overlap':s}
lib.chkfile.save(name+'_integrals.h5', 'molecule', atom_dic)

