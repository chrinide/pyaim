#!/usr/bin/env python

import os
import sys
import time
import h5py
import numpy
import ctypes
import signal
from pyscf import lib
from pyscf import dft
from pyscf.lib import logger

signal.signal(signal.SIGINT, signal.SIG_DFL)

_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)
libcgto = lib.load_library('libcgto')

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

# Very simple asume equal l in both atoms
def multipolar(self):
    idx = self.inucs[0]
    jdx = self.inucs[1]
    rint = numpy.zeros(3)
    rint = self.xyzrho[jdx] - self.xyzrho[idx]
    lmax = self.lmax[0]
    NPROPS = lmax*(lmax+2) + 1
    coeff = numpy.zeros((NPROPS,NPROPS))
    coulm = numpy.zeros((NPROPS,NPROPS))
    jlm = numpy.zeros((NPROPS),dtype=numpy.int32)
    coulp = numpy.zeros(lmax+1)
    feval = 'eval_gaunt'
    drv = getattr(libaim, feval)
    drv(ctypes.c_int(lmax), 
        rint.ctypes.data_as(ctypes.c_void_p),
        coeff.ctypes.data_as(ctypes.c_void_p), 
        jlm.ctypes.data_as(ctypes.c_void_p))
    coul = 0.0
    for lm1 in range(0,NPROPS):
        for lm2 in range(0,NPROPS):
            q1 = self.qlm1[lm1]
            q2 = self.qlm2[lm2]
            coulm[lm1,lm2] = q1*q2*coeff[lm2,lm1]
            coul += coulm[lm1,lm2]
    logger.info(self,"Total Multipolar Coulomb interaction %f" % coul)
    for lm1 in range(0,NPROPS):
        l1 = jlm[lm1]
        for lm2 in range(0,NPROPS):
            l2 = jlm[lm2]
            l = max(l1,l2)
            coulp[l] += coulm[lm1,lm2]
    for l in range(1,lmax+1):
        coulp[l] = coulp[l-1]+coulp[l]
    logger.info(self,"Multipolar Coulomb interaction in L")
    for l in range(0,lmax+1):
        logger.info(self,"L %d %f" % (l,coulp[l]))


class Multi(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inucs = [0,1]
##################################################
# don't modify the following attributes, they are not input options
        self.mol = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.nelectron = None
        self.xyzrho = None
        self.charge = None
        self.spin = None
        self.lmax = numpy.empty(2, dtype=numpy.int32)
        self.qlm1 = None
        self.qlm2 = None
        self._keys = set(self.__dict__.keys())

    def dump_input(self):

        if self.verbose < logger.INFO:
            return self

        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'* General Info')
        logger.info(self,'Date %s' % time.ctime())
        logger.info(self,'Python %s' % sys.version)
        logger.info(self,'Numpy %s' % numpy.__version__)
        logger.info(self,'Number of threads %d' % self.nthreads)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'Max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

        logger.info(self,'* Molecular Info')
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])

        logger.info(self,'* Multipolar interaction info')
        logger.info(self,'Properties for nucs %d %d', *self.inucs)
        logger.info(self,'Lmax for each nuc %d %d', *self.lmax)
        logger.info(self,'')

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 3

        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.nelectron = self.mol.nelectron 
        self.charge = self.mol.charge    
        self.spin = self.mol.spin      
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()

        idx = 'atom'+str(self.inucs[0])
        with h5py.File(self.surfile) as f:
            self.xyzrho = f[idx+'/xyzrho'].value
        idx = 'qlm'+str(self.inucs[0])
        jdx = 'qlm'+str(self.inucs[0])
        with h5py.File(self.surfile) as f:
            self.lmax[0] = f[idx+'/lmax'].value
            self.qlm1 = f[idx+'/totprops'].value
            self.lmax[1] = f[jdx+'/lmax'].value
            self.qlm2 = f[jdx+'/totprops'].value
    
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        multipolar(self)
        logger.info(self,'Write info to HDF5 file')

        logger.info(self,'')
        logger.info(self,'Coulomb Multipolar of atoms %d %d done', *self.inucs)
        logger.timer(self,'Multipolar build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'h2o.chk'
    bas = Multi(name)
    bas.verbose = 4
    bas.inucs = [0,1]
    bas.kernel()

