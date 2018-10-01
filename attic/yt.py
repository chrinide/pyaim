#!/usr/bin/env python

import numpy
import time
from pyscf import gto, scf, lib
from pyscf.dft import numint, gen_grid
    
mol = gto.M()
mol.atom = '''O  0.000000  0.000000  0.000000 
              H  0.761561  0.478993  0.000000 
              H -0.761561  0.478993  0.000000'''
mol.basis = 'aug-cc-pvqz'
mol.verbose = 4
mol.symmetry = 1
mol.build()

mf = scf.RHF(mol)
mf.scf()
dm = mf.make_rdm1()

nx = 101
ny = 101
nz = 101

coord = mol.atom_coords()
box = numpy.max(coord,axis=0) - numpy.min(coord,axis=0) + 6.0
boxorig = numpy.min(coord,axis=0) - 3.0
# .../(nx-1) to get symmetric mesh
xs = numpy.arange(nx) * (box[0] / (nx-1))
ys = numpy.arange(ny) * (box[1] / (ny-1))
zs = numpy.arange(nz) * (box[2] / (nz-1))
coords = lib.cartesian_prod([xs,ys,zs])
coords = numpy.asarray(coords, order='C') - (-boxorig)

ngrids = nx * ny * nz
blksize = min(8000, ngrids)
rho = numpy.empty(ngrids)
ao = None
for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
    ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
    rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
rho = numpy.sort(rho, kind='mergesort')
