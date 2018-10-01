#!/usr/bin/env python

import os
import sys
import time
import numpy
from pyscf import dft
from pyscf import lib
from pyscf.lib import logger

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

class MoleQ(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.scratch = lib.param.TMPDIR 
##################################################
# don't modify the following attributes, they are not input options
        self.mol = None
        self.mo_coeff = None
        self.mo_occ = None
        self.natm = None
        self.coords = None
        self.charges = None
        self._keys = set(self.__dict__.keys())

    def dump_input(self):
        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.mol.nelectron)
        logger.info(self,'Total charge %d' % self.mol.charge)
        logger.info(self,'Spin %d ' % self.mol.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %8.5f, %8.5f, %8.5f', i, 
                        self.charges[i], *self.coords[i])
        return self

    def build(self):
        t0 = time.clock()
        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        if self.verbose > logger.NOTE:
            logger.TIMER_LEVEL = self.verbose
            self.dump_input()
        logger.timer(self,'MoleQ build', t0)

        return self
    kernel = build

if __name__ == '__main__':
    name = 'h2o.chk'
    mol = MoleQ(name)
    mol.verbose = 4
    mol.build()

