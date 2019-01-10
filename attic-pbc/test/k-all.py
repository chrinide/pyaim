#!/usr/bin/env python

import numpy
from pyscf.pbc import gto, scf, dft, df
from pyscf import lib
import pyscf.lib.parameters as param
from pyscf.pbc.tools import pbc

name = 'k-diamond'

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
    basis = '6-31g',
    verbose = 4,
)

nk = [4,4,4]  # 4 k-poins for each axis, 4^3=64 kpts in total
kpts = cell.make_kpts(nk)
scf.chkfile.save_cell(cell, name+'.chk')

supercell = [1,1,1]
super_cell = pbc.super_cell(cell, supercell)
lattice = super_cell.lattice_vectors() * param.BOHR
symbols = [atom[0] for atom in super_cell._atom]
cart = numpy.asarray([(numpy.asarray(atom[1])* param.BOHR).tolist() for atom in super_cell._atom])
num_atoms = super_cell.natm		
with open(name + '_struct.xsf', 'w') as f:
	f.write('Generated by pyscf\n\n')		
	f.write('CRYSTAL\n')
	f.write('PRIMVEC\n')	
	for row in range(3):
		f.write('%10.7f  %10.7f  %10.7f\n' % (lattice[0, row], lattice[1, row], lattice[2, row]))	
	f.write('PRIMVEC\n')
	for row in range(3):
		f.write('%10.7f  %10.7f  %10.7f\n' % (lattice[0, row], lattice[1, row], lattice[2, row]))	
	f.write('PRIMCOORD\n')
	f.write('%3d %3d\n' % (num_atoms, 1))
	for atom in range(len(symbols)):
		f.write('%s  %7.7f  %7.7f  %7.7f\n' % (symbols[atom], cart[atom][0], cart[atom][1], cart[atom][2]))				
	f.write('\n\n')			

nks = [4,4,4]
kpts = cell.make_kpts(nks)
#kpts -= kpts[0] # Shift to gamma

#kmdf = df.FFTDF(cell, kpts)
#kmf = dft.KRKS(cell, kpts)
kmf = dft.KRKS(cell, kpts).density_fit(auxbasis='weigend')
kmf.chkfile = name+'.chk'
#kmf.init_guess = 'chk'
#kmf.with_df = kmdf
kmf.with_df._cderi_to_save = name+'.h5'
kmf.xc = 'bp86'
kmf.kernel()

