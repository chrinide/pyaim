#!/usr/bin/env python

from pyscf import gto, scf

mol = gto.M(
    atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
    charge = 0,
    spin = 4,
    verbose = 4
)

mf = scf.GHF(mol)
dm = mf.get_init_guess() + 0j
mf.kernel(dm0=dm)
print mf.mo_coeff.dtype
                      
