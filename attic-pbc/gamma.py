#!/usr/bin/env python

import numpy
from pyscf import lib, scf, gto, ao2mo
from pyscf.pbc import df  as pbcdf
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
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

kpts = [0,0,0]
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

mf = pbcscf.RHF(cell).density_fit(auxbasis='def2-svp-jkfit')
mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
#mf.init_guess = 'chk'
#mf.with_df._cderi_to_save = name+'.h5'
mf.with_df._cderi = name+'.h5'
#mf.with_df.mesh = [10,10,10] # Tune PWs in MDF for performance/accuracy balance
mf = scf.addons.remove_linear_dep_(mf)
mf.kernel()
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

