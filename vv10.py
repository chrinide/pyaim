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

def rho_grad(self,x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(self.mol, x, deriv=2)
    ngrids, nao = ao[0].shape
    pos = self.mo_occ > self.occdrop
    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros((5,ngrids))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    for i in range(1, 4):
        c1 = numpy.dot(ao[i], cpos)
        rho[i] += numpy.einsum('pi,pi->p', c0, c1)*2.0
    rho[4] = numpy.einsum('pi,pi->i',rho[-3:],rho[-3:])
    rho[4] = numpy.sqrt(rho[4])
    rho[4] *= rho[4]
    return rho

def vv10_e(self):
    t0 = time.time()
    rho1 = rho_grad(self, self.p1)
    rho2 = rho_grad(self, self.p2)
    logger.info(self,'Time precomputing grid values %.3f (sec)' % (time.time()-t0))
    t0 = time.time()
    ex = dft.libxc.eval_xc('rPW86,', rho1)[0]
    ec = dft.libxc.eval_xc(',PBE', rho1)[0]
    ex = numpy.dot(rho1[0], self.w1*ex)
    ec = numpy.dot(rho1[0], self.w1*ec)
    rhoat = numpy.dot(rho1[0], self.w1)
    logger.info(self,'Atom %d Rho, X, C %f %f %f' % (self.inucs[0],rhoat,ex,ec))
    ex = dft.libxc.eval_xc('rPW86,', rho2)[0]
    ec = dft.libxc.eval_xc(',PBE', rho2)[0]
    ex = numpy.dot(rho2[0], self.w2*ex)
    ec = numpy.dot(rho2[0], self.w2*ec)
    rhoat = numpy.dot(rho2[0], self.w2)
    logger.info(self,'Atom %d Rho, X, C %f %f %f' % (self.inucs[1],rhoat,ex,ec))
    logger.info(self,'Time for some checks %.3f (sec)' % (time.time()-t0))
    libaim.vv10.restype = ctypes.c_double
    ev = libaim.vv10(ctypes.c_int(self.npoints[0]),
                     ctypes.c_int(self.npoints[1]),
                     ctypes.c_double(self.coef_C),
                     ctypes.c_double(self.coef_B),
                     self.p1.ctypes.data_as(ctypes.c_void_p),
                     rho1[0].ctypes.data_as(ctypes.c_void_p),
                     rho1[4].ctypes.data_as(ctypes.c_void_p),
                     self.w1.ctypes.data_as(ctypes.c_void_p),
                     self.p2.ctypes.data_as(ctypes.c_void_p),
                     rho2[0].ctypes.data_as(ctypes.c_void_p),
                     rho2[4].ctypes.data_as(ctypes.c_void_p),
                     self.w2.ctypes.data_as(ctypes.c_void_p))
    return ev

class VV10(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inucs = [0,0]
        self.occdrop = 1e-6
        self.coef_C = 0.0093
        self.coef_B = 5.9
##################################################
# don't modify the following attributes, they are not input options
        self.mol = None
        self.mo_coeff = None
        self.mo_occ = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.npoints= None
        self.w1 = None
        self.p1 = None
        self.w2 = None
        self.p2 = None
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

        logger.info(self,'* Grid and VV10 Info')
        logger.info(self,'Properties for nucs %d %d', *self.inucs)
        logger.info(self,'Npoints %d %d', *self.npoints)
        logger.info(self,'Coef C %f' % self.coef_C)
        logger.info(self,'Coef B %f' % self.coef_B)
        logger.info(self,'')

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 3

        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        self.nelectron = self.mol.nelectron 
        self.charge = self.mol.charge    
        self.spin = self.mol.spin      
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()

        idx = 'grid'+str(self.inucs[0])
        jdx = 'grid'+str(self.inucs[1])
        with h5py.File(self.surfile) as f:
            self.p1 = f[idx+'/p'].value
            self.w1 = f[idx+'/w'].value
            self.p2 = f[jdx+'/p'].value
            self.w2 = f[jdx+'/w'].value
        self.npoints = numpy.zeros(2, dtype=numpy.int32)
        self.npoints[0] = len(self.w1)
        self.npoints[1] = len(self.w2)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        with lib.with_omp_threads(self.nthreads):
            ev = vv10_e(self)
        lib.logger.info(self,'VV10 energy %f' % ev)
        logger.info(self,'Write info to HDF5 file')

        logger.info(self,'VV10 of atoms %d %d done', *self.inucs)
        logger.timer(self,'VV10 build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'h2o.chk'
    bas = VV10(name)
    bas.verbose = 4
    bas.inucs = [0,0]
    bas.kernel()

