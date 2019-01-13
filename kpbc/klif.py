#!/usr/bin/env python

import numpy
from pyscf import scf as ascf
from pyscf.pbc import gto, scf, dft, lib

name = 'klif'

cell = gto.M(
    a = numpy.eye(3)*4.0834274,
    atom = '''
           Li      0.0000000   0.0000000   0.0000000
           Li      0.0000000   2.0417137   2.0417137
           Li      2.0417137   0.0000000   2.0417137
           Li      2.0417137   2.0417137   0.0000000
           F       0.0000000   0.0000000   2.0417137
           F       2.0417137   2.0417137   2.0417137
           F       2.0417137   0.0000000   0.0000000
           F       0.0000000   2.0417137   0.0000000
           ''',
    basis = 'def2-svp',
    verbose = 4,
)

nk = [2,2,2]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
kpts -= kpts[0] # Shift to gamma
scf.chkfile.save_cell(cell, name+'.chk')
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

kmf = dft.KRKS(cell, kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf = ascf.addons.remove_linear_dep_(kmf)
#kmf.grids = dft.gen_grid.BeckeGrids(cell)
#kmf.grids.level = 4
#kmf.grids.prune = None
kmf.xc = 'pbe,pbe'
kmf.chkfile = name+'.chk'
kmf.with_df._cderi_to_save = name+'.h5'
#kmf.with_df._cderi = name+'.h5'
kmf.kernel()

########################
nkpts = len(kpts)
def point(r):
    ao = dft.numint.eval_ao_kpts(cell, r, kpts=kpts, deriv=1)
    rhograd = numpy.zeros((4,1))
    for k in range(nkpts):
        rhograd += dft.numint.eval_rho2(cell, ao[k], kmf.mo_coeff[k], kmf.mo_occ[k], xctype='GGA')
    rhograd *= 1./nkpts
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


