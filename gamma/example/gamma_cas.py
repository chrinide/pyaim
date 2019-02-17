#!/usr/bin/env python

import numpy, avas
from pyscf.pbc import gto, scf
from pyscf import fci, ao2mo, mcscf, lib
from pyscf import scf as mole_scf
from pyscf.tools import wfn_format

name = 'gamma_cas'

cell = gto.Cell()
cell.incore_anyway = True
cell.atom='''
  H 0.000000000000   0.000000000000   0.000000000000
  H 1.000000000000   0.000000000000   0.000000000000
'''
cell.basis = 'def2-svp'
cell.precision = 1e-12
cell.dimension = 1
cell.a = [[2,0,0],[0,1,0],[0,0,1]]
cell.unit = 'A'
cell.verbose = 4
cell.build()

mf = scf.RHF(cell).density_fit()
mf.with_df.auxbasis = 'def2-svp-jkfit'
mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.with_df._cderi_to_save = name+'_eri.h5'
#mf.with_df._cderi = name+'_eri.h5' 
#mf = mole_scf.addons.remove_linear_dep_(mf)
ehf = mf.kernel()

kpts = [0,0,0]
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

ao_labels = ['H 1s']
ncas, nelecas, mo = avas.avas(mf, ao_labels, ncore=0, minao='ano', with_iao=True, \
                                             threshold_occ=0.1, threshold_vir=0.1)

mc = mcscf.CASSCF(mf, ncas, nelecas)
#mc.fix_spin_(shift=.4, ss=0)
mc.fcisolver = fci.direct_spin0.FCI()
#mc.fcisolver = fci.selected_ci_spin0.SCI()
#mc.fcisolver.ci_coeff_cutoff = 0.005
#mc.fcisolver.select_cutoff = 0.005
mc.kernel(mo)

nmo = mc.ncore + mc.ncas
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)
lib.logger.info(mf,'Write rdm1-rdm2 on MO basis to HDF5 file')
dic = {'rdm1':rdm1,
       'rdm2':rdm2}
lib.chkfile.save(name+'.chk', 'pdm', dic)

natocc, natorb = numpy.linalg.eigh(-rdm1)
for i, k in enumerate(numpy.argmax(abs(natorb), axis=0)):
    if natorb[k,i] < 0:
        natorb[:,i] *= -1
natorb = numpy.dot(mc.mo_coeff[:,:nmo], natorb)
natocc = -natocc

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, cell, natorb, mo_occ=natocc)
    wfn_format.write_coeff(f2, cell, mc.mo_coeff[:,:nmo])
    wfn_format.write_ci(f2, mc.ci, mc.ncas, mc.nelecas, ncore=mc.ncore)

#c = mc.mo_coeff[:,:nmo]
#s = cell.pbc_intor('cint1e_ovlp_sph')
#t = cell.pbc_intor('cint1e_kin_sph')
#h = mc.get_hcore()
#s = reduce(numpy.dot, (c.T,s,c))
#t = reduce(numpy.dot, (c.T,t,c))
#h = reduce(numpy.dot, (c.T,h,c))
#enuc = cell.energy_nuc() 
#ekin = numpy.einsum('ij,ji->',t,rdm1)
#hcore = numpy.einsum('ij,ji->',h,rdm1)
#pop = numpy.einsum('ij,ji->',s,rdm1)
#print('Population : %s' % pop)
#print('Kinetic energy : %s' % ekin)
#print('Hcore energy : %s' % hcore)
#print('Nuclear energy : %s' % enuc)

#eri_mo = ao2mo.kernel(mf._eri, c, compact=False)
#eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
#rdm2_j = lib.einsum('ij,kl->ijkl', rdm1, rdm1) 
#rdm2_xc = rdm2 - rdm2_j
#bie1 = numpy.einsum('ijkl,ijkl->', eri_mo, rdm2_j)*0.5 
#bie2 = numpy.einsum('ijkl,ijkl->', eri_mo, rdm2_xc)*0.5
#print('J energy : %s' % bie1)
#print('XC energy : %s' % bie2)
#print('EE energy : %s' % (bie1+bie2))

#etot = enuc + hcore + bie1 + bie2
#print('Total energy : %s' % etot)

nmo = mc.mo_coeff.shape[1]
rdm1, rdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas) 
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(rdm1, rdm2, mc.ncore, mc.ncas, nmo)
lib.logger.info(mf,'Write rdm1 on MO basis to HDF5 file')
dic = {'rdm1':rdm1}
lib.chkfile.save(name+'.chk', 'rdm', dic)

