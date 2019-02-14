#!/usr/bin/env python

import numpy, avas
from pyscf import gto, scf, lib, mcscf
from pyscf.tools import wfn_format

name = 'lih'

mol = gto.Mole()
mol.atom = '''
Li      0.000000      0.000000      0.389516
H      0.000000      0.000000     -1.209516
'''
mol.basis = 'def2-svp'
mol.verbose = 4
mol.spin = 0
mol.symmetry = 0
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

wfn_file = name + '.wfn'
idx = mf.mo_occ>0
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,idx], mo_occ=mf.mo_occ[idx], mo_energy=mf.mo_energy[idx])

aolst1  = ['Li 2s', 'H 1s']
aolst2  = ['Li 2p']
aolst = aolst1 + aolst2
ncas, nelecas, mo = avas.avas(mf, aolst, threshold_occ=0.1, threshold_vir=1e-5, minao='ano', ncore=1)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.chkfile = name+'.chk'
mc.max_cycle_macro = 35
mc.max_cycle_micro = 7
mc.kernel(mo)

nmo = mc.mo_coeff.shape[1]#mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)

natocc, natorb = numpy.linalg.eigh(-rdm1)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc

wfn_file = name + '_cas.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, mol, mc.mo_coeff[:,:nmo])
    wfn_format.write_ci(f2, mc.ci, mc.ncas, mc.nelecas, ncore=mc.ncore)

lib.logger.info(mf,'Write rdm1 on MO basis to HDF5 file')
dic = {'rdm1':rdm1}
lib.chkfile.save(name+'.chk', 'rdm', dic)

s = mol.intor('int1e_ovlp')
t = mol.intor('int1e_kin')
coeff = mc.mo_coeff#[:,:nmo]
s = coeff.T.dot(s).dot(coeff)
t = coeff.T.dot(t).dot(coeff)

ekin = numpy.einsum('ij,ji->',t,rdm1)
pop = numpy.einsum('ij,ji->',s,rdm1)
lib.logger.info(mf,'Population : %12.6f' % pop)
lib.logger.info(mf,'Kinetic energy : %12.6f' % ekin)

