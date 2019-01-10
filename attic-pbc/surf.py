#!/usr/bin/env python

import os
import sys
import time
import numpy
import ctypes
import signal

from pyscf.pbc import lib as libpbc
from pyscf import lib
from pyscf.lib import logger

import grid

libpbcgto = lib.load_library('libpbc')
_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

GRADEPS = 1e-10
RHOEPS = 1e-10
MINSTEP = 1e-6
MAXSTEP = 0.75
SAFETY = 0.9
ENLARGE = 1.6
HMINIMAL = numpy.finfo(numpy.float64).eps

def rhograd(self, x):

    x = numpy.reshape(x, (-1,3))
    x = numpy.asarray(x, order='F')
    kpts = numpy.reshape(self.kpts, (-1,3))
    if self.cart:
        feval = 'PBCGTOval_cart_deriv1'
    else:
        feval = 'PBCGTOval_sph_deriv1' 
    kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)
    out = numpy.empty((nkpts,4,self.nao,1), dtype=numpy.complex128)
    expLk = self.explk
    rcut = self.rcut

    drv = getattr(libpbcgto, feval)
    drv(ctypes.c_int(1),
        (ctypes.c_int*2)(*self.shls_slice), self.ao_loc.ctypes.data_as(ctypes.c_void_p),
        self.ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(self.ls)),
        expLk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts),
        out.ctypes.data_as(ctypes.c_void_p),
        x.ctypes.data_as(ctypes.c_void_p),
        rcut.ctypes.data_as(ctypes.c_void_p),
        self.non0tab.ctypes.data_as(ctypes.c_void_p),
        self.atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.natm),
        self.bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.nbas),
        self.env.ctypes.data_as(ctypes.c_void_p))

    ao_kpts = []
    for k, kpt in enumerate(kpts_lst):
        v = out[k]
        if abs(kpt).sum() < 1e-9:
            v = numpy.asarray(v.real, order='C')
        v = v.transpose(0,2,1)
        ao_kpts.append(v)

    rho = numpy.empty((4,1))
    if nkpts == 1:  # A single k-point
        ao_kpts = ao_kpts[0]
        pos = self.mo_occ > self.occdrop
        cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
        c0 = numpy.dot(ao_kpts[0], cpos)
        rho[0] = numpy.einsum('pi,pi->p',c0, c0)
        c1 = numpy.dot(ao_kpts[1], cpos)
        rho[1] = numpy.einsum('pi,pi->p', c0, c1)*2
        c1 = numpy.dot(ao_kpts[2], cpos)
        rho[2] = numpy.einsum('pi,pi->p', c0, c1)*2
        c1 = numpy.dot(ao_kpts[3], cpos)
        rho[3] = numpy.einsum('pi,pi->p', c0, c1)*2
        gradmod = numpy.linalg.norm(rho[-3:,0])
    else:
        pass

    return rho[0,0], rho[-3:,0]/(gradmod+HMINIMAL), gradmod

def gradrho(self, xpoint, h):

    h0 = h
    niter = 0
    rho, grad, gradmod = rhograd(self,xpoint)
    grdt = grad
    grdmodule = gradmod

    while (grdmodule > GRADEPS and niter < self.mstep):
        niter += 1
        ier = 1
        while (ier != 0):
            xtemp = xpoint + h0*grdt
            rho, grad, gradmod = rhograd(self,xtemp)
            escalar = numpy.einsum('i,i->',grdt,grad) 
            if (escalar < 0.707):
                if (h0 >= MINSTEP):
                    h0 = h0/2.0
                    ier = 1
                else:
                    ier = 0
            else:
                if (escalar > 0.9): 
                    hproo = h0*ENLARGE
                    if (hproo < h):
                        h0 = hproo
                    else:
                        h0 = h
                    h0 = numpy.minimum(MAXSTEP, h0)
                ier = 0
                xpoint = xtemp
                grdt = grad
                grdmodule = gradmod
            logger.debug(self,'scalar, step in gradrho %.6f %.6f', escalar, h0)

    logger.debug(self,'nsteps in gradrho %d', niter)

    return xpoint, grdmodule

EXTRA_PREC = 1e-2
def _estimate_rcut(self):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    log_prec = numpy.log(self.cell.precision * EXTRA_PREC)
    rcut = []
    for ib in range(self.nbas):
        l = self.cell.bas_angular(ib)
        es = self.cell.bas_exp(ib)
        cs = abs(self.cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        rcut.append(r.max())
    return numpy.array(rcut)

class BaderSurf(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inuc = 0
        self.epsiscp = 0.180
        self.ntrial = 11
        self.leb = True
        self.npang = 5810
        self.nptheta = 90
        self.npphi = 180
        self.iqudt = 'legendre'
        self.rmaxsurf = 10.0
        self.rprimer = 0.4
        self.backend = 'rkck'
        self.epsroot = 1e-5
        self.epsilon = 1e-5 
        self.step = 0.1
        self.mstep = 120
        self.corr = False
        self.occdrop = 1e-6
##################################################
# don't modify the following attributes, they are not input options
        self.cell = None
        self.a = None
        self.mo_coeff = None
        self.mo_occ = None
        self.nocc = None
        self.kpts = None
        self.nkpts = None
        self.ls = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.xnuc = None
        self.xyzrho = None
        self.rpru = None
        self.grids = None
        self.rsurf = None
        self.nlimsurf = None
        self.rmin = None
        self.rmax = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.atm = None
        self.bas = None
        self.nbas = None
        self.nprims = None
        self.nmo = None
        self.env = None
        self.ao_loc = None
        self.shls_slice = None
        self.nao = None
        self.non0tab = None
        self.cart = None
        self.rdm1 = None
        self.rcut = None
        self.explk = None
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

        logger.info(self,'* Cell Info')
        for i in range(3):
            logger.info(self,'Cell %d axis : %.6f  %.6f  %.6f', i, *self.a[i])

        logger.info(self,'Number of cells %d' % len(self.ls))
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])

        logger.info(self,'* Basis Info')
        logger.info(self,'Is cartesian %s' % self.cart)
        logger.info(self,'Number of molecular orbitals %d' % self.nmo)
        logger.info(self,'Orbital EPS occ criterion %e' % self.occdrop)
        #logger.info(self,'Number of occupied molecular orbitals %d' % self.nocc)
        logger.info(self,'Number of molecular primitives %d' % self.nprims)
        logger.debug(self,'Occs : %s' % self.mo_occ) 

        logger.info(self,'* Surface Info')
        if (self.leb):
            logger.info(self,'Lebedev quadrature')
        else:
            logger.info(self,'Theta quadrature %s' % self.iqudt)
            logger.info(self,'Phi is always trapezoidal')
            logger.info(self,'N(theta,phi) points %d %d' % (self.nptheta,self.npphi))
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'Surface for nuc %d' % self.inuc)
        logger.info(self,'Rmaxsurface %f' % self.rmaxsurf)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Rprimer %f' % self.rprimer)
        logger.debug(self,'Rpru : %s' % self.rpru) 
        logger.info(self,'Epsiscp %f' % self.epsiscp)
        logger.info(self,'Epsroot %e' % self.epsroot)
        logger.info(self,'ODE solver %s' % self.backend)
        logger.info(self,'ODE tool %e' % self.epsilon)
        logger.info(self,'Max steps in ODE solver %d' % self.mstep)
        logger.info(self,'')

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 3

        cell = libpbc.chkfile.load_cell(self.chkfile)
        cell.ecp = None
        self.cell = cell
        self.a = numpy.array(cell.a)
        self.nelectron = cell.nelectron 
        self.charge = cell.charge    
        self.spin = cell.spin      
        self.natm = cell.natm		
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        #self.kpts = lib.chkfile.load(self.chkfile, 'scf/kpts')
        self.kpts = [0,0,0]
        self.nkpts = 1
        self.ls = cell.get_lattice_Ls(dimension=3)
        self.ls = self.ls[numpy.argsort(lib.norm(self.ls, axis=1))]
        self.atm = numpy.asarray(cell._atm, dtype=numpy.int32, order='C')
        self.bas = numpy.asarray(cell._bas, dtype=numpy.int32, order='C')
        self.env = numpy.asarray(cell._env, dtype=numpy.double, order='C')
        self.nbas = self.bas.shape[0]
        self.ao_loc = cell.ao_loc_nr()
        self.shls_slice = (0, self.nbas)
        sh0, sh1 = self.shls_slice
        self.nao = self.ao_loc[sh1] - self.ao_loc[sh0]
        self.non0tab = numpy.empty((1,self.nbas), dtype=numpy.int8)
        # non0tab stores the number of images to be summed in real space.
        # Initializing it to 255 means all images are summed
        self.non0tab[:] = 0xff
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in cell._atom])
        self.charges = cell.atom_charges()
        nprims, nmo = self.mo_coeff.shape 
        self.nprims = nprims
        self.nmo = nmo
        self.cart = cell.cart
        if (not self.leb):
            self.npang = self.npphi*self.nptheta

        self.rcut = _estimate_rcut(self)
        kpts = numpy.reshape(self.kpts, (-1,3))
        kpts_lst = numpy.reshape(kpts, (-1,3))
        self.explk = numpy.exp(1j * numpy.asarray(numpy.dot(self.ls, kpts_lst.T), order='C'))

        if (self.ntrial%2 == 0): self.ntrial += 1
        geofac = numpy.power(((self.rmaxsurf-0.1)/self.rprimer),(1.0/(self.ntrial-1.0)))
        self.rpru = numpy.zeros((self.ntrial))
        for i in range(self.ntrial): 
            self.rpru[i] = self.rprimer*numpy.power(geofac,(i+1)-1)
        self.rsurf = numpy.zeros((self.npang,self.ntrial), order='C')
        self.nlimsurf = numpy.zeros((self.npang), dtype=numpy.int32)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        if (self.iqudt == 'legendre'):
            self.iqudt = 1
        
        if (self.leb):
            self.grids = grid.lebgrid(self.npang)
        else:
            self.grids = grid.anggrid(self.iqudt,self.nptheta,self.npphi)
         
        self.xyzrho = numpy.zeros((self.natm,3))
        t = time.time()
        for i in range(self.natm):
            self.xyzrho[i], gradmod = gradrho(self,self.coords[i],self.step)
            if (gradmod > 1e-4):
                if (self.charges[i] > 2.0):
                    logger.info(self,'Good rho position %.6f %.6f %.6f', *self.xyzrho[i])
                else:
                    raise RuntimeError('Failed finding nucleus:', *self.xyzrho[i]) 
            else:
                logger.info(self,'Check rho position %.6f %.6f %.6f', *self.xyzrho[i])
                logger.info(self,'Setting xyzrho for atom to imput coords')
                self.xyzrho[i] = self.coords[i]
        self.xnuc = numpy.asarray(self.xyzrho[self.inuc])
        logger.info(self,'Time finding nucleus %.3f (sec)' % (time.time()-t))

        if (self.backend == 'rkck'):
            backend = 1
        elif (self.backend == 'rkdp'):
            backend = 2
        else:
            raise NotImplementedError('Only rkck or rkdp ODE solver yet available') 
         
        ct_ = numpy.asarray(self.grids[:,0], order='C')
        st_ = numpy.asarray(self.grids[:,1], order='C')
        cp_ = numpy.asarray(self.grids[:,2], order='C')
        sp_ = numpy.asarray(self.grids[:,3], order='C')
        
        t = time.time()
        feval = 'surf_driver'
        drv = getattr(libaim, feval)
        with lib.with_omp_threads(self.nthreads):
            drv(ctypes.c_int(self.inuc), 
                self.xyzrho.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.npang), 
                ct_.ctypes.data_as(ctypes.c_void_p),
                st_.ctypes.data_as(ctypes.c_void_p),
                cp_.ctypes.data_as(ctypes.c_void_p),
                sp_.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.ntrial), 
                self.rpru.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_double(self.epsiscp), 
                ctypes.c_double(self.epsroot), 
                ctypes.c_double(self.rmaxsurf), 
                ctypes.c_int(backend),
                ctypes.c_double(self.epsilon), 
                ctypes.c_double(self.step), 
                ctypes.c_int(self.mstep),
                ctypes.c_int(self.natm), 
                self.coords.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.cart),
                ctypes.c_int(self.nmo),  
                ctypes.c_int(self.nprims), 
                self.atm.ctypes.data_as(ctypes.c_void_p), 
                ctypes.c_int(self.nbas), 
                self.bas.ctypes.data_as(ctypes.c_void_p), 
                self.env.ctypes.data_as(ctypes.c_void_p), 
                self.ao_loc.ctypes.data_as(ctypes.c_void_p),
                self.mo_coeff.ctypes.data_as(ctypes.c_void_p),
                self.mo_occ.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_double(self.occdrop), 
                #
                self.a.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(len(self.ls)), 
                self.ls.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(self.nkpts), 
                self.explk.ctypes.data_as(ctypes.c_void_p),
                self.rcut.ctypes.data_as(ctypes.c_void_p),
                self.non0tab.ctypes.data_as(ctypes.c_void_p),
                #
                self.nlimsurf.ctypes.data_as(ctypes.c_void_p),
                self.rsurf.ctypes.data_as(ctypes.c_void_p))
        logger.info(self,'Time finding surface %.3f (sec)' % (time.time()-t))
             
        self.rmin = 1000.0
        self.rmax = 0.0
        for i in range(self.npang):
            nsurf = int(self.nlimsurf[i])
            self.rmin = numpy.minimum(self.rmin,self.rsurf[i,0])
            self.rmax = numpy.maximum(self.rmax,self.rsurf[i,nsurf-1])
        logger.info(self,'Rmin for surface %.6f', self.rmin)
        logger.info(self,'Rmax for surface %.6f', self.rmax)

        logger.info(self,'Write HDF5 surface file')
        atom_dic = {'inuc':self.inuc,
                    'xnuc':self.xnuc,
                    'xyzrho':self.xyzrho,
                    'coords':self.grids,
                    'npang':self.npang,
                    'ntrial':self.ntrial,
                    'rmin':self.rmin,
                    'rmax':self.rmax,
                    'nlimsurf':self.nlimsurf,
                    'rsurf':self.rsurf}
        lib.chkfile.save(self.surfile, 'atom'+str(self.inuc), atom_dic)
        logger.info(self,'Surface of atom %d saved',self.inuc)
        logger.timer(self,'BaderSurf build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'gamma.chk'
    surf = BaderSurf(name)
    surf.epsilon = 1e-5
    surf.epsroot = 1e-5
    surf.verbose = 4
    surf.epsiscp = 0.320
    surf.mstep = 300
    surf.npang = 5810
    surf.inuc = 0
    surf.kernel()

