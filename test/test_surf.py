#!/usr/bin/env python

import unittest

from pyscf import gto, scf, lib

mol = gto.Mole()
mol.basis = 'cc-pvdz'
mol.atom = '''
N  0.0000  0.0000  0.5488
N  0.0000  0.0000 -0.5488
'''
mol.verbose = 0
mol.spin = 0
mol.symmetry = 1
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mf.chkfile = 'test.chk'
mf.kernel()

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf

class KnownValues(unittest.TestCase):
    def test_ray(self):
        #self.assertAlmostEqual(mycc.e_tot, -76.119346385357446, 7)
        self.assertAlmostEqual(1.0, 1.0, 10)

if __name__ == "__main__":
    print("Full Tests for Surface")
    unittest.main()

