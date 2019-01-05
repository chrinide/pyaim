#!/usr/bin/env python

import numpy, os
from pyscf import gto, scf, lib, dft
from pyscf.tools import wfn_format

name = 'co'

mol = gto.Mole()
mol.atom = '''
C      0.000000      0.000000     -0.642367
O      0.000000      0.000000      0.481196
'''
dirnow = os.path.realpath(os.path.join(__file__, '..'))
basfile = os.path.join(dirnow, 'sqzp.dat')
mol.basis = basfile
mol.verbose = 4
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = dft.RKS(mol)
mf.xc = 'wb97x'
mf.grids.prune = None
mf.grids.level = 5
mf.max_cycle = 150
mf.chkfile = name+'.chk'
mf.kernel()
dm = mf.make_rdm1()

wfn_file = name + '.wfn'
with open(wfn_file, 'w') as f2:
    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], \
    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])

unit = 1.0
charges = mol.atom_charges()
coords  = mol.atom_coords()
charge_center = numpy.einsum('i,ij->j', charges, coords)/charges.sum()
origin = charge_center
mol.set_common_orig(origin)
lib.logger.info(mf,'Setting origing to charge center: %.4f, %.4f, %.4f', *origin)

ao_dip = mol.intor_symmetric('int1e_r', comp=3)
el_dip = numpy.einsum('xij,ji->x', ao_dip, dm)
lib.logger.info(mf,'Electronic Dipole moment(X, Y, Z, Au): %.4f, %.4f, %.4f', *el_dip*unit)
nucl_dip = numpy.einsum('i,ix->x', charges, coords)
lib.logger.info(mf,'Nuclear Dipole moment(X, Y, Z, Au): %.4f, %.4f, %.4f', *nucl_dip*unit)
mol_dip = (nucl_dip - el_dip) * unit
lib.logger.info(mf,'Total Dipole moment(X, Y, Z, Au): %.4f, %.4f, %.4f', *mol_dip)

