#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf, dft, lib

name = 'mo-kpts'

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''C     0.      0.      0.    
              C     0.8917  0.8917  0.8917
              C     1.7834  1.7834  0.    
              C     2.6751  2.6751  0.8917
              C     1.7834  0.      1.7834
              C     2.6751  0.8917  2.6751
              C     0.      1.7834  1.7834
              C     0.8917  2.6751  2.6751''',
    basis = 'def2-svp',
    verbose = 4,
)

nk = [2,2,2]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
kpts -= kpts[0] # Shift to gamma
scf.chkfile.save_cell(cell, name+'.chk')
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)
nkpts = len(kpts)
weight = 1.0/len(kpts)
print('The k-weights %s' % weight)

kmf = dft.KRKS(cell, kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.grids = dft.gen_grid.BeckeGrids(cell)
kmf.grids.level = 3
kmf.xc = 'pbe,pbe'
kmf.chkfile = name+'.chk'
#kmf.with_df._cderi_to_save = name+'.h5'
kmf.with_df._cderi = 'kpts.h5'
kmf.kernel()
dm = kmf.make_rdm1()

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

