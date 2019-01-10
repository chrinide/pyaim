#!/usr/bin/env python

import numpy
from pyscf.pbc import df as pbcdf
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.tools import pyscf_ase
from pyscf.pbc.tools import lattice
from pyscf.pbc import tools
from pyscf.pbc.tools.pbc import super_cell
from pyscf import mcscf, lo, lib, ao2mo
from pyscf.tools import wfn_format
from pyscf.mcscf import avas
from pyscf.lo import orth
einsum = lib.einsum

nmp = [2, 2, 2]
ase_atom = lattice.get_ase_alkali_halide(A='Li', B='F')
cell = pbcgto.Cell()
cell.spin = 0
cell.symmetry = 0
cell.charge = 0
cell.unit = 'B'
cell.verbose = 4
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.build()

new = super_cell(cell, nmp)
cell = pbcgto.Cell()
#cell.unit = 'B'
cell.spin = 0
cell.symmetry = 0
cell.charge = 0
cell.verbose = 4
cell.atom = new.atom
cell.a = new.a
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.mesh = [10,10,10]
cell.build()

mf = pbchf.RHF(cell)
mf.with_df = pbcdf.AFTDF(cell)
mf.kernel()

nao = cell.nao_nr()
dm = mf.make_rdm1()
mo = mf.mo_coeff
c = numpy.eye(cell.nao_nr())
s = cell.pbc_intor('cint1e_ovlp_sph')

pop = einsum('ij,ji->',s,dm)
print('Population : %s' % pop)

pairs1 = einsum('ij,kl,ij,kl->',dm,dm,s,s)*0.5 # J
pairs2 = einsum('ij,kl,li,kj->',dm,dm,s,s)*0.25 # XC
pairs = (pairs1 - pairs2)
print('Coulomb Pairs : %12.6f' % (pairs1))
print('XC Pairs : %12.6f' % (pairs2))
print('Pairs : %12.6f' % pairs)

##############################################################################
##############################################################################
label = cell.spheric_labels(False)
print('Monocentric terms')
print('#################')
pop = einsum('ij,ji->i', dm,s)
chg1 = numpy.zeros(cell.natm)
qq1 = numpy.zeros(cell.natm)
for i, s1 in enumerate(label):
    chg1[s1[0]] += pop[i]
for ia in range(cell.natm):
    symb = cell.atom_symbol(ia)
    qq1[ia] = cell.atom_charge(ia)-chg1[ia]
    print('Pop, Q, of %d %s = %12.6f %12.6f' % (ia, symb, chg1[ia], qq1[ia]))

print('\nSum rules test')
print('##############')
print('Sum of charges : %12.6f' % sum(chg1))

print('\nBicentric terms')
print('###############')
pairs1 = einsum('ij,kl,ij,kl->ik',dm,dm,s,s)*0.5 # J
pairs2 = einsum('ij,kl,li,kj->ik',dm,dm,s,s)*0.25 # XC
pop = (pairs1 - pairs2)
chg = numpy.zeros((cell.natm,cell.natm))
chg1 = numpy.zeros((cell.natm,cell.natm))
chg2 = numpy.zeros((cell.natm,cell.natm))
for i, s1 in enumerate(label):
    for j, s2 in enumerate(label):
        factor = 1.0
        chg[s1[0],s2[0]] += pop[i,j]*factor
        chg1[s1[0],s2[0]] += pairs1[i,j]*factor
        chg2[s1[0],s2[0]] += pairs2[i,j]*factor
check = 0        
checkii = 0
checkij = 0
checkj = 0
checkxc = 0
for ia in range(cell.natm):
    symb1 = cell.atom_symbol(ia)
    for ib in range(ia+1):
        symb2 = cell.atom_symbol(ib)
        if (ia == ib): 
            factor = 1.0
            checkii = checkii + chg[ia,ib]
            check = check + chg[ia,ib]
            checkj = checkj + chg1[ia,ib]
            checkxc = checkxc + chg2[ia,ib]
        if (ia != ib): 
            factor = 2.0
            checkj = checkj + factor*chg1[ia,ib]
            checkxc = checkxc + factor*chg2[ia,ib]
            checkij = checkij + factor*chg[ia,ib]
            check = check + factor*chg[ia,ib]
        print('Lambda-Delta, Pairs of  %d %d %s %s = %12.6f %12.6f ' % (ia, ib, symb1, symb2, 2*factor*chg2[ia,ib], chg[ia,ib]))

print('\n##############')
print('Sum rules test')
print('##############')
print('Total Coulomb pairs : %12.6f' % checkj)
print('Total XC pairs : %12.6f' % checkxc)
print('Total intra pairs : %12.6f' % checkii)
print('Total inter pairs : %12.6f' % checkij)
print('Total pairs : %12.6f' % check)
