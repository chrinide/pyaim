#!/usr/bin/env python

from pyscf import gto, scf

mol = gto.M(
    atom = '''
O 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587''',
    basis = 'ccpvdz',
    charge = 1,
    spin = 1,
    verbose = 4
)

mf = scf.GHF(mol)
dm = mf.get_init_guess() + 0j
dm[0,:] += .1j
dm[:,0] -= .1j
mf.kernel(dm0=dm)
                      
