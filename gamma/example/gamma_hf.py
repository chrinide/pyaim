#!/usr/bin/env python

import numpy, avas
from pyscf.pbc import gto, scf
from pyscf import fci, ao2mo, mcscf, lib
from pyscf import scf as mole_scf
from pyscf.tools import wfn_format

name = 'gamma_hf'

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

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, cell, mf.mo_coeff[:,mf.mo_occ>0], \
    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])

