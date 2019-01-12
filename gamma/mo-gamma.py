#!/usr/bin/env python

import numpy
from pyscf import lib, scf, gto, ao2mo
from pyscf.pbc import df  as pbcdf
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
einsum = lib.einsum

name = 'mo-gamma'

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

kpts = [0,0,0]
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

mf = pbcscf.RHF(cell).density_fit(auxbasis='def2-svp-jkfit')
mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
#mf.init_guess = 'chk'
#mf.with_df._cderi_to_save = name+'.h5'
mf.with_df._cderi = 'gamma.h5'
#mf.with_df.mesh = [10,10,10] # Tune PWs in MDF for performance/accuracy balance
mf = scf.addons.remove_linear_dep_(mf)
mf.kernel()

c = mf.mo_coeff[:,mf.mo_occ>0]
nmo = c.shape[1]
occ = mf.mo_occ[mf.mo_occ>0]
dm = numpy.diag(occ)

##############################################################################
# Get momo electronic integrals
##############################################################################
s = cell.pbc_intor('cint1e_ovlp_sph')
t = cell.pbc_intor('cint1e_kin_sph')
h = mf.get_hcore()
s = reduce(numpy.dot, (c.T,s,c))
t = reduce(numpy.dot, (c.T,t,c))
h = reduce(numpy.dot, (c.T,h,c))
##############################################################################
enuc = cell.energy_nuc() 
ekin = numpy.einsum('ij,ij->',t,dm)
hcore = numpy.einsum('ij,ij->',h,dm)
pop = numpy.einsum('ij,ij->',s,dm)
print('Population : %s' % pop)
print('Kinetic energy : %s' % ekin)
print('Hcore energy : %s' % hcore)
print('Nuclear energy : %s' % enuc)
##############################################################################

##############################################################################
# Get bielectronic integrals
##############################################################################
eri_mo = ao2mo.kernel(mf._eri, c, compact=False)
eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
##############################################################################
rdm2_j = einsum('ij,kl->ijkl', dm, dm) 
rdm2_xc = -0.5*einsum('il,kj->ilkj', dm, dm)
rdm2 = rdm2_j + rdm2_xc
bie1 = einsum('ijkl,ijkl->', eri_mo, rdm2_j)*0.5 
bie2 = einsum('ijkl,ilkj->', eri_mo, rdm2_xc)*0.5
print('J energy : %s' % bie1)
print('XC energy : %s' % bie2)
print('EE energy : %s' % (bie1+bie2))
##############################################################################

etot = enuc + hcore + bie1 + bie2
print('Total energy : %s' % etot)

dm = mf.make_rdm1()
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

