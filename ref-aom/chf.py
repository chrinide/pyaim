#!/usr/bin/env python

import numpy
from functools import reduce
from pyscf import gto, scf, lo, dft, lib
from pyscf.tools import wfn_format, molden

def get_pop(dm, s, lmultipliers):

    label = mol.spheric_labels(False)

    #pop = numpy.einsum('ij,ji->i', dm,s)
    #chg1 = numpy.zeros(mol.natm)
    #qq1 = numpy.zeros(mol.natm)
    #for i, s1 in enumerate(label):
    #    chg1[s1[0]] += pop[i]
    #for ia in range(mol.natm):
    #    symb = mol.atom_symbol(ia)
    #    qq1[ia] = mol.atom_charge(ia)-chg1[ia]
    #    lib.logger.info(mf, 'Pop, Q, of %d %s = %12.6f %12.6f' \
    #    % (ia+1, symb, chg1[ia], qq1[ia]))

    nao = dm.shape[0]
    fock = numpy.zeros_like(dm)
    for ia in range(mol.natm):
        g = numpy.zeros_like(dm)
        iden = numpy.zeros_like(dm)
        for i, s1 in enumerate(label):
            if (s1[0]==ia):
                iden[i,:] = 1.0
        iden = 0.5*(iden+iden.T)
        print iden
        #tmp = numpy.einsum('ij,ji->ij',iden,s)
        #tmp = numpy.einsum('ij,ji->',tmp,dm)
        #lib.logger.info(mf, 'Pop of %d = %12.6f' % (ia+1, tmp))
        g = numpy.einsum('ij,ji->ij', iden,s)
        fock += lmultipliers[ia]*g

    return fock

def get_cfock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
    fock = old_get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
    fock -= get_pop(dm, s1e, multiplicadores) 
    return fock

mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
Li  0.000000000000000  0.000000000000000  0.000000000000000
H   0.000000000000000  0.000000000000000  1.563900000000000
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = 0
mol.build()

mf = scf.RHF(mol)
mf.verbose = 0
mf.kernel()
dm = mf.make_rdm1()

multiplicadores = numpy.zeros(mol.natm)

my_diis_obj = scf.ADIIS()
my_diis_obj.diis_space = 14

x = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0.0,
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x = numpy.asarray(x)
ehf = []

#for b in x:
#    multiplicadores = [0.0,b]

multiplicadores = [0.0,0.1]
mf = scf.RHF(mol)
mf.conv_tol = 1e-6
mf.max_cycle = 120
mf.diis = my_diis_obj
old_get_fock = mf.get_fock
mf.get_fock = get_cfock 
#mf.verbose = 3
ehf.append(mf.kernel(dm0=dm))
mf.analyze(with_meta_lowdin=False)
dm = mf.make_rdm1()

#name = 'lif_0p6'    
#wfn_file = name + '.wfn'
#with open(wfn_file, 'w') as f2:
#    wfn_format.write_mo(f2, mol, mf.mo_coeff[:,mf.mo_occ>0], \
#    mo_occ=mf.mo_occ[mf.mo_occ>0], mo_energy=mf.mo_energy[mf.mo_occ>0])

