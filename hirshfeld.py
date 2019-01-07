#!/usr/bin/env python

import os
import sys
import time
import h5py
import numpy
import ctypes
import signal

from pyscf import lib, dft
from pyscf.lib import logger
from pyscf.lib.parameters import BOHR

# Free Atomic Polarizabilities from
# CRC Handbook of Chemistry and Physics, 88th Ed.
# untis are AA^3, then transformed to au (0.529...)^3.
FREPOL = (1.0/BOHR**3) * numpy.array((0,
    0.6668,   0.2051,  24.3300,   5.6000,   3.0300,   1.7600, 
    1.1000,   0.8020,   0.5570,   0.3956,  24.1100,  10.6000, 
    6.8000,   5.3800,   3.6300,   2.9000,   2.1800,   1.6411, 
   43.4000,  22.8000,  17.8000,  14.6000,  12.4000,  11.6000, 
    9.4000,   8.4000,   7.5000,   6.8000,   6.2000,   5.7500, 
    8.1200,   6.0700,   4.3100,   3.7700,   3.0500,   2.4844, 
   47.3000,  27.6000,  22.7000,  17.9000,  15.7000,  12.8000, 
   11.4000,   9.6000,   8.6000,   4.8000,   7.2000,   7.3600, 
   10.2000,   7.7000,   6.6000,   5.5000,   5.3500,   4.0440, 
   59.4200,  39.7000,  31.1000,  29.6000,  28.2000,  31.4000, 
   30.1000,  28.8000,  27.7000,  23.5000,  25.5000,  24.5000, 
   23.6000,  22.7000,  21.8000,  21.0000,  21.9000,  16.2000, 
   13.1000,  11.1000,   9.7000,   8.5000,   7.6000,   6.5000, 
    5.8000,   5.0200,   7.6000,   6.8000,   7.4000,   6.8000, 
    6.0000,   5.3000,  48.6000,  38.3000,  32.1000,  32.1000, 
   25.4000,  24.9000,  24.8000,  24.5000,  23.3000,  23.0000, 
   22.7000,  20.5000,  19.7000,  23.8000,  18.2000,  17.5000)) 

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)
libcgto = lib.load_library('libcgto')

HMINIMAL = numpy.finfo(numpy.float64).eps

# TODO: screaning of points
def rho(self,x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(self.mol, x, deriv=2)
    ngrids, nao = ao[0].shape
    pos = self.mo_occ > self.occdrop
    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros((6,ngrids))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    for i in range(1, 4):
        c1 = numpy.dot(ao[i], cpos)
        rho[i] += numpy.einsum('pi,pi->p', c0, c1)*2.0
        rho[4] += numpy.einsum('pi,pi->p', c1, c1)
    XX, YY, ZZ = 4, 7, 9
    ao2 = ao[XX] + ao[YY] + ao[ZZ]
    c1 = numpy.dot(ao2, cpos)
    rho[5] = numpy.einsum('pi,pi->p', c0, c1)
    rho[5] += rho[4]
    rho[5] *= 2
    rho[4] *= 0.5
    return rho

#def prange(start, end, step):
#    for i in range(start, end, step):
#        yield i, min(i+step, end)
#blksize = min(8000, ngrids)
#buffer = numpy.empty(ngrids)
#for ip0, ip1 in lib.prange(0, ngrids, blksize):
#    ao = numint.eval_ao(mol, coords[ip0:ip1])
#    buffer[ip0:ip1] = numpy.dot(ao, coeff)
#def prune_small_rho_grids(self):
#    rhop = rho(self,self.p)
#    rhop *= self.w
#    idx = abs(rhop) > self.small_rho_cutoff/self.w.size
#    logger.info(self,'Dropped grids %d' % (self.w.size - numpy.count_nonzero(idx)))
#    self.p = numpy.asarray(self.p[idx], order='C')
#    self.w = numpy.asarray(self.w[idx], order='C')
#    return self

def vfree(self):    
    self.frevol = numpy.zeros(self.natm)
    chf = self.chf
    libaim.frevol.restype = ctypes.c_double
    for i in range(self.natm):
        self.frevol[i] = libaim.frevol(ctypes.c_int(self.charges[i]), ctypes.c_double(chf))
    return self

def integrate(self):
    rhop = rho(self,self.grids.coords)
    rhoval = numpy.dot(rhop[0],self.grids.weights)
    logger.info(self,'Integral of rho %f' % rhoval)
    ftmp = setweights(self)
    atomq = numpy.zeros(self.natm)
    atomk = numpy.zeros(self.natm)
    atoml = numpy.zeros(self.natm)
    npoints = len(self.grids.weights)
    hirshfeld = numpy.zeros(npoints)
    for i in range(self.natm):
        hirshfeld[:] = ftmp['weight'+str(i)]
        atomq[i] = numpy.dot(rhop[0],self.grids.weights*hirshfeld)
        atomk[i] = numpy.dot(rhop[4],self.grids.weights*hirshfeld)
        atoml[i] = numpy.dot(rhop[5],self.grids.weights*hirshfeld)
        logger.info(self,'Q,K,L of atom %d %f %f %f' % (i,atomq[i],atomk[i],atoml[i]))
    return self

def setweights(self):
    logger.info(self,'Getting atomic data and weigths from tabulated densities')
    npoints = len(self.grids.weights)
    output = numpy.zeros(npoints)
    promol = numpy.zeros(npoints)
    ftmp = lib.H5TmpFile()
    rhoat = 0.0
    for i in range(self.natm):
        libaim.eval_atomic(ctypes.c_int(npoints),  
                           ctypes.c_int(self.charges[i]),  
                           self.coords[i].ctypes.data_as(ctypes.c_void_p), 
                           self.grids.coords.ctypes.data_as(ctypes.c_void_p), 
                           output.ctypes.data_as(ctypes.c_void_p))
        h5dat = ftmp.create_dataset('atom'+str(i), (npoints,), 'f8')
        h5dat[:] = output[:]
        rhoa = numpy.einsum('i,i->',output,self.grids.weights)
        rhoat += rhoa
        promol += output
        logger.info(self,'Integral of rho for atom %d %f' % (i, rhoa))
    logger.info(self,'Integral of rho promolecular %f' % rhoat)
    for i in range(self.natm):
        h5dat = ftmp.create_dataset('weight'+str(i), (npoints,), 'f8')
        h5dat[:] = ftmp['atom'+str(i)][:]/(promol+HMINIMAL)
    return ftmp                

class Hirshfeld(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.non0tab = False
        self.corr = False
        self.occdrop = 1e-6
        self.chf = 0.0
        self.small_rho_cutoff = 1e-6
        self.prune = False
##################################################
# don't modify the following attributes, they are not input options
        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.grids = dft.Grids(self.mol)
        self.frevol = None
        self.rdm1 = None
        self.nocc = None
        self.mo_coeff = None
        self.mo_occ = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.nprims = None
        self.nmo = None
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
        logger.info(self,'Correlated ? %s' % self.corr)

        logger.info(self,'* Molecular Info')
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d with Vfree %f and position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], self.frevol[i], *self.coords[i])

        logger.info(self,'* Basis Info')
        logger.info(self,'Number of molecular orbitals %d' % self.nmo)
        logger.info(self,'Orbital EPS occ criterion %e' % self.occdrop)
        logger.info(self,'Number of occupied molecular orbitals %d' % self.nocc)
        logger.info(self,'Number of molecular primitives %d' % self.nprims)
        logger.debug(self,'Occs : %s' % self.mo_occ) 

        logger.info(self,'* Grid Info')
        logger.info(self,'Pruning grids %s', self.grids.prune)
        if self.grids.atom_grid:
            logger.info(self,'User specified grid scheme %s', str(self.grids.atom_grid))
        else:
            logger.info(self,'Grids dens level %d', self.grids.level)
        logger.info(self,'Number of points %s', len(self.grids.weights))
        logger.info(self,'Radial grids %s', self.grids.radi_method)
        logger.info(self,'Becke partition %s', self.grids.becke_scheme)
        if self.grids.radii_adjust is not None:
            logger.info(self,'Atomic radii adjust function %s',
                        self.grids.radii_adjust)
            logger.debug(self,'Atomic_radii %s', self.grids.atomic_radii)
        logger.info(self,'')

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 3

        self.nelectron = self.mol.nelectron 
        self.charge = self.mol.charge    
        self.spin = self.mol.spin      
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        nprims, nmo = self.mo_coeff.shape 
        self.nprims = nprims
        self.nmo = nmo

        if (self.corr):
            self.rdm1 = lib.chkfile.load(self.chkfile, 'rdm/rdm1') 
            natocc, natorb = numpy.linalg.eigh(self.rdm1)
            natorb = numpy.dot(self.mo_coeff, natorb)
            self.mo_coeff = natorb
            self.mo_occ = natocc
        nocc = self.mo_occ[abs(self.mo_occ)>self.occdrop]
        nocc = len(nocc)
        self.nocc = nocc

        self.grids.verbose = 0
        self.grids.stdout = self.stdout
        self.grids.build()

        vfree(self)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        with lib.with_omp_threads(self.nthreads):
            integrate(self)

        logger.info(self,'Hirshfeld properties done')
        logger.timer(self,'Hirshfeld build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'h2o.chk'
    bas = Hirshfeld(name)
    bas.verbose = 4
    bas.grids.level = 4
    bas.grids.prune = None
    bas.kernel()

