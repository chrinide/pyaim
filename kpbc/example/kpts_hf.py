#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf
from pyscf import lib
from pyscf import scf as mole_scf

name = 'kpts_hf'

cell = gto.Cell()
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

nk = [1,1,1]
kpts = cell.make_kpts(nk)
kpts -= kpts[0] # Shift to gamma
scf.chkfile.save_cell(cell, name+'.chk')
dic = {'kpts':kpts}
lib.chkfile.save(name+'.chk', 'kcell', dic)

mf = scf.KRHF(cell, kpts).density_fit()
mf.with_df.auxbasis = 'def2-svp-jkfit'
#mf.exxdiv = None
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.with_df._cderi_to_save = name+'_eri.h5'
#mf.with_df._cderi = name+'_eri.h5' 
#mf = mole_scf.addons.remove_linear_dep_(mf)
ehf = mf.kernel()

