#!/usr/bin/env python

import numpy
from pyscf import lib, scf, gto, ao2mo
from pyscf.pbc import df  as pbcdf
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.tools import wfn_format
einsum = lib.einsum

name = 'gamma'

cell = pbcgto.Cell()
cell.atom = '''C     0.      0.      0.    
               C     0.8917  0.8917  0.8917
               C     1.7834  1.7834  0.    
               C     2.6751  2.6751  0.8917
               C     1.7834  0.      1.7834
               C     2.6751  0.8917  2.6751
               C     0.      1.7834  1.7834
               C     0.8917  2.6751  2.6751'''
cell.a = numpy.eye(3)*3.5668
cell.basis = 'def2-svp'
cell.verbose = 4
cell.symmetry = 0
cell.build()

mf = pbcscf.RHF(cell).mix_density_fit(auxbasis='def2-svp-jkfit')
mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
#mf.init_guess = 'chk'
mf.with_df._cderi_to_save = name+'.h5'
#mf.with_df._cderi = name+'.h5'
mf.with_df.mesh = [10,10,10] # Tune PWs in MDF for performance/accuracy balance
mf = scf.addons.remove_linear_dep_(mf)
mf.kernel()

wfn_file = name+'.wfn'
fspt = open(wfn_file,'w')
coeff = mf.mo_coeff[:,mf.mo_occ>0]
occ = mf.mo_occ[mf.mo_occ>0]
energy = mf.mo_energy[mf.mo_occ>0]
a = cell.a
t = cell.get_lattice_Ls()
t = t[numpy.argsort(lib.norm(t, axis=1))]
kpts = numpy.asarray([0.0, 0.0, 0.0])
#expLk = numpy.exp(1j * numpy.asarray(numpy.dot(t, kpts.T), order='C'))

wfn_format.write_mo(fspt, cell, coeff, mo_energy=energy, mo_occ=occ)
fspt.write('CRYSTAL\n')
fspt.write('CELL\n')
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[0][0], a[0][1], a[0][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[1][0], a[1][1], a[1][2]))
fspt.write(' %11.8f %11.8f %11.8f\n' % (a[2][0], a[2][1], a[2][2]))
fspt.write('N-KPOINTS 1\n')
fspt.write(' %11.8f %11.8f %11.8f\n' % (kpts[0], kpts[1], kpts[2]))
fspt.write('T-VECTORS %3d\n' % len(t))
for i in range(len(t)):
    fspt.write(' %11.8f %11.8f %11.8f\n' % (t[i][0], t[i][1], t[i][2]))
fspt.close()

dm = mf.make_rdm1()

s = cell.pbc_intor('int1e_ovlp')
k = cell.pbc_intor('int1e_kin')
v = cell.pbc_intor('int1e_nuc')
enuc = cell.energy_nuc() 
ekin = einsum('ij,ji->',k,dm)
pop = einsum('ij,ji->',s,dm)
print('Population : %12.6f' % pop)
print('Kinetic energy : %12.6f' % ekin)
print('Nuclear Repulsion energy : %12.6f' % enuc)

core = mf.get_hcore()
core = einsum('ij,ji->',core,dm)
print('1e-energy : %12.6f' % core)

nao = cell.nao_nr()
eri_ao = ao2mo.restore(1, mf._eri, nao)
eri_ao = eri_ao.reshape(nao,nao,nao,nao)
bie1 = einsum('ijkl,ij,kl',eri_ao,dm,dm)*0.5 # J
bie2 = numpy.einsum('ijkl,il,jk',eri_ao,dm,dm)*0.25 # XC
pairs1 = einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = numpy.einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
print('Coulomb Pairs : %12.6f' % (pairs1))
print('XC Pairs : %12.6f' % (pairs2))
print('Pairs : %12.6f' % pairs)
print('J energy : %12.6f' % bie1)
print('XC energy : %12.6f' % -bie2)
print('EE energy : %12.6f' % (bie1-bie2))

vj, vk = mf.with_df.get_jk(dm, exxdiv=mf.exxdiv)
print('J',einsum('ij,ji->', vj, dm)*0.5)
print('K',einsum('ij,ji->', vk, dm)*0.25)

etot = enuc + core + bie1 - bie2
print('Total energy : %12.6f' % etot)

########################
def point(r):
    ao = pbcdft.numint.eval_ao(cell, r, deriv=1)
    rhograd = pbcdft.numint.eval_rho(cell, ao, dm, xctype='GGA') 
    grad = numpy.array([rhograd[1], rhograd[2], rhograd[3]], dtype=numpy.float64)
    return rhograd[0], grad

print "################# 0.0"
r = numpy.array([0.00000, 0.00000, 0.00000]) 
r = numpy.reshape(r, (-1,3))
rho, grad = point(r)
print r, rho, grad
print "################# 0.5"
r = numpy.array([0.50000, 0.00000, 0.00000]) 
r = numpy.reshape(r, (-1,3))
rho, grad = point(r)
print r, rho, grad
print "################# 1.0"
r = numpy.array([1.00000, 0.00000, 0.00000]) 
r = numpy.reshape(r, (-1,3))
rho, grad = point(r)
print r, rho, grad
print "################# 1.8"
r = numpy.array([1.80000, 0.00000, 0.00000]) 
r = numpy.reshape(r, (-1,3))
rho, grad = point(r)
print r, rho, grad
print "################# 3.6"
r = numpy.array([3.60000, 0.00000, 0.00000]) 
r = numpy.reshape(r, (-1,3))
rho, grad = point(r)
print r, rho, grad
