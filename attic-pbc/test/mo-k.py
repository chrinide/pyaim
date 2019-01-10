#!/usr/bin/env python

# TODO : HF with 2-RDM en MO basis, I need ERIS in MO basis

import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pbcscf
import pyscf.pbc.df  as pbcdf
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import tools

cell = pbcgto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.mesh = [10,10,10] 
cell.build()
print('The cell volume = %s' % cell.vol)

nks = [2,2,2] 
kpts = cell.make_kpts(nks)
nkpts = len(kpts)
weight = 1.0/len(kpts)
print('The k-weights = %s' % weight)

pdf = pbcdf.FFTDF(cell, kpts)

kmf = pbcscf.KRHF(cell, kpts)
kmf.with_df = pdf
kmf.exxdiv = None
kmf.max_cycle = 150
kmf.kernel()
dm = kmf.make_rdm1()
madelung = tools.pbc.madelung(cell, kpts)
print('Madelung constant = %s' % madelung)

##############################################################################
# Get momo electronic integrals
##############################################################################
s = cell.pbc_intor('cint1e_ovlp_sph', kpts=kpts)
t = cell.pbc_intor('cint1e_kin_sph', kpts=kpts)
h = kmf.get_hcore()
##############################################################################
enuc = cell.energy_nuc() 
hcore = 1.0/nkpts * numpy.einsum('kij,kji->', dm, h)
ekin = 1.0/nkpts * numpy.einsum('kij,kji->', dm, t)
pop = 1.0/nkpts * numpy.einsum('kij,kji->', dm, s)
print('Population : %s' % pop)
print('Kinetic AO energy : %s' % ekin)
print('Hcore AO energy : %s' % hcore)
print('Nuclear energy : %s' % enuc)
##############################################################################

###############################################################################
## Get effective J,K two electron potential matrix
###############################################################################
vj, vk = kmf.get_jk()
bie1 = 1.0/nkpts * numpy.einsum('kij,kji', dm, vj) * 0.5
bie2 = 1.0/nkpts * numpy.einsum('kij,kji', dm, vk) * 0.25
print('J energy : %s' % bie1)
print('XC energy : %s' % -bie2)
print('EE energy : %s' % (bie1-bie2))
###############################################################################
 
###############################################################################
etot = enuc + hcore + bie1 - bie2
print('Total energy : %s' % etot)
###############################################################################

##############################################################################
# Mono electronic integrals in the MO basis
##############################################################################
hcore = 0.0
ekin = 0.0
for ki in range(nkpts):
    coeff = kmf.mo_coeff[ki][:,kmf.mo_occ[ki]>0]
    occ = kmf.mo_occ[ki][kmf.mo_occ[ki]>0]
    rdm1 = numpy.zeros((len(occ),len(occ)),dtype=dm.dtype)
    rdm1 = numpy.diag(occ)
    h_ki = reduce(numpy.dot, (coeff.conj().T,h[ki],coeff))
    t_ki = reduce(numpy.dot, (coeff.conj().T,t[ki],coeff))
    hcore += numpy.einsum('ij,ji->',h_ki,rdm1)
    ekin += numpy.einsum('ij,ji->',t_ki,rdm1)
hcore /= nkpts
ekin /= nkpts
print('Kinetic MO energy : %s' % ekin)
print('Hcore MO energy : %s' % hcore)

