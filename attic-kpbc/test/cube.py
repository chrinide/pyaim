#!/usr/bin/env python

import numpy
import time
import pyscf
import h5py
import pyscf.lib.parameters as param
from pyscf.scf import hf
from pyscf.pbc import gto, scf, dft, df
from pyscf.pbc import lib as libpbc
from pyscf.pbc.tools import pbc
from pyscf import lib

name = 'k-lif'
cell = libpbc.chkfile.load_cell(name+'.chk')
cell.ecp = None
mo_coeff_kpts = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ_kpts = lib.chkfile.load(name+'.chk', 'scf/mo_occ')
kpts = lib.chkfile.load(name+'.chk', 'scf/kpts')
nkpts = len(mo_occ_kpts)
dm_kpts = [hf.make_rdm1(mo_coeff_kpts[k], mo_occ_kpts[k]) for k in range(nkpts)]
dm_kpts = lib.asarray(dm_kpts)

supercell = [1,1,1]
grid = [50,50,50]
ngrid = numpy.asarray(grid)
nx, ny, nz = numpy.asarray(grid)
ngrids = nx * ny * nz

xs = 0.1
ys = 0.1
zs = 0.1

super_cell = pbc.super_cell(cell, supercell)
lattice = super_cell.lattice_vectors()
symbols = [atom[0] for atom in super_cell._atom]
cart = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in super_cell._atom])
num_atoms = super_cell.natm		

qv = lib.cartesian_prod([numpy.arange(i) for i in ngrid])
a_frac = numpy.einsum('i,ij->ij', 1./(ngrid - 1), cell.lattice_vectors())
coords = numpy.dot(qv, a_frac)
coords = numpy.asarray(coords, order='C')

ao = None
ao = dft.numint.eval_ao_kpts(cell, coords, kpts=kpts, deriv=0)
rho = numpy.zeros(ngrids)
for k in range(nkpts):
    rho += dft.numint.eval_rho2(cell, ao[k], mo_coeff_kpts[k], mo_occ_kpts[k], xctype='LDA')
rho *= 1./nkpts
rho = rho.reshape(nx,ny,nz)

with open(name + '_den.cube', 'w') as f:
    f.write('Electron density in real space (e/Bohr^3)\n')
    f.write('PySCF Version: %s  Date: %s\n' % (pyscf.__version__, time.ctime()))
    f.write('%5d' % num_atoms)
    f.write('%12.6f%12.6f%12.6f\n' % tuple(numpy.zeros(3).tolist()))
    f.write('%5d%12.6f%12.6f%12.6f\n' % (nx, xs[1], 0, 0))
    f.write('%5d%12.6f%12.6f%12.6f\n' % (ny, 0, ys[1], 0))
    f.write('%5d%12.6f%12.6f%12.6f\n' % (nz, 0, 0, zs[1]))
    for ia in range(mol.natm):
        chg = super_cell.atom_charge(ia)
        f.write('%5d%12.6f'% (chg, chg))
        f.write('%12.6f%12.6f%12.6f\n' % (cart[atom][0], cart[atom][1], cart[atom][2]))
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(0,nz,6):
                remainder  = (nz-iz)
                if (remainder > 6 ):
                    fmt = '%13.5E' * 6 + '\n'
                    f.write(fmt % tuple(rho[ix,iy,iz:iz+6].tolist()))
                else:
                    fmt = '%13.5E' * remainder + '\n'
                    f.write(fmt % tuple(rho[ix,iy,iz:iz+remainder].tolist()))
                    break

