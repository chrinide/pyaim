#!/usr/bin/env python

import sys
import time
import h5py
import numpy
import signal
from pyscf import lib
from pyscf import dft
from pyscf.lib import logger

import grid

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

EPS = 1e-7

def rho(self,x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(self.mol, x, deriv=0)
    ngrids, nao = ao.shape
    pos = self.mo_occ > self.occdrop
    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros(ngrids)
    c0 = numpy.dot(ao, cpos)
    rho = numpy.einsum('pi,pi->p', c0, c0)
    return rho

def mom(self,x,w,origin):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(self.mol, x, deriv=0)
    ngrids, nao = ao.shape
    pos = self.mo_occ > self.occdrop
    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros(ngrids)
    c0 = numpy.dot(ao, cpos)
    rho = numpy.einsum('pi,pi->p', c0, c0)
    r = numpy.zeros(ngrids)
    for i in range(ngrids):
        r[i] = numpy.linalg.norm(x[i,:]-origin)
    mom1 = numpy.dot(rho,r*w)
    mom2 = numpy.dot(rho,r*r*w) 
    mom3 = numpy.dot(rho,r*r*r*w) 
    return mom1,mom2,mom3

def prune_small_rho_grids(self):
    rhop = rho(self,self.p)
    rhop *= self.w
    idx = abs(rhop) > self.small_rho_cutoff/self.w.size
    logger.info(self,'Dropped grids %d' % (self.w.size - numpy.count_nonzero(idx)))
    self.p = numpy.asarray(self.p[idx], order='C')
    self.w = numpy.asarray(self.w[idx], order='C')
    return self

def inbasin(self,r,j):
    isin = False
    rs1 = 0.0
    irange = self.nlimsurf[j]
    for k in range(irange):
        rs2 = self.rsurf[j,k]
        if (r >= rs1-EPS and r <= rs2+EPS):
            if (((k+1)%2) == 0):
                isin = False
            else:
                isin = True
            return isin
        rs1 = rs2
    return isin

def out_beta(self):
    logger.info(self,'* Go outside betasphere')
    xcoor = numpy.zeros(3)
    nrad = self.nrad
    npang = self.npang
    iqudr = self.iqudr
    mapr = self.mapr
    r0 = self.brad
    rfar = self.rmax
    rad = self.rad
    t0 = time.time()
    rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
    coordsang = self.agrids
    coords = []
    weigths = []
    for n in range(nrad):
        r = rmesh[n]
        for j in range(npang):
            inside = True
            inside = inbasin(self,r,j)
            if (inside == True):
                cost = coordsang[j,0]
                sintcosp = coordsang[j,1]*coordsang[j,2]
                sintsinp = coordsang[j,1]*coordsang[j,3]
                xcoor[0] = r*sintcosp
                xcoor[1] = r*sintsinp
                xcoor[2] = r*cost    
                p = self.xnuc + xcoor
                coords.append(p)
                weigths.append(coordsang[j,4]*dvol[n]*rwei[n])
    coords = numpy.array(coords)
    weigths = numpy.array(weigths)
    npoints = len(weigths)
    logger.info(self,'Outside number of points %d' % npoints)
    logger.info(self,'Time out Bsphere %.3f (sec)' % (time.time()-t0))
    return coords, weigths
    
def int_beta(self): 
    logger.info(self,'* Go with inside betasphere')
    xcoor = numpy.zeros(3)
    npang = self.bnpang
    coords = numpy.empty((npang,3))
    nrad = self.bnrad
    iqudr = self.biqudr
    mapr = self.bmapr
    r0 = 0
    rfar = self.brad
    rad = self.rad
    t0 = time.time()
    rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
    coordsang = grid.lebgrid(npang)
    coords = []
    weigths = []
    for n in range(nrad):
        r = rmesh[n]
        for j in range(npang): # j-loop can be changed to map
            cost = coordsang[j,0]
            sintcosp = coordsang[j,1]*coordsang[j,2]
            sintsinp = coordsang[j,1]*coordsang[j,3]
            xcoor[0] = r*sintcosp
            xcoor[1] = r*sintsinp
            xcoor[2] = r*cost    
            p = self.xnuc + xcoor
            coords.append(p)
            weigths.append(coordsang[j,4]*dvol[n]*rwei[n])
    coords = numpy.asarray(coords)
    weigths = numpy.asarray(weigths)
    npoints = len(weigths)
    logger.info(self,'Inside number of points %d' % npoints)
    logger.info(self,'Time in Bsphere %.3f (sec)' % (time.time()-t0))
    return coords, weigths

class Points(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
        self.inuc = 0
        self.nrad = 101
        self.iqudr = 'legendre'
        self.mapr = 'becke'
        self.betafac = 0.4
        self.bnrad = 101
        self.bnpang = 3074
        self.biqudr = 'legendre'
        self.bmapr = 'becke'
        self.occdrop = 1e-6
        self.small_rho_cutoff = 1e-6
        self.prune = False
##################################################
# don't modify the following attributes, they are not input options
        self.mol = None
        self.mo_coeff = None
        self.mo_occ = None
        self.ntrial = None
        self.npang = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.xnuc = None
        self.xyzrho = None
        self.agrids = None
        self.rsurf = None
        self.nlimsurf = None
        self.rmin = None
        self.rmax = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.rad = None
        self.brad = None
        self.w = None
        self.p = None
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

        logger.info(self,'* Surface Info')
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'Properties for nuc %d' % self.inuc)
        logger.info(self,'Nuclear coordinate %.6f  %.6f  %.6f', *self.xnuc)
        logger.info(self,'Rho nuclear coordinate %.6f  %.6f  %.6f', *self.xyzrho[self.inuc])
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Rmin for surface %f', self.rmin)
        logger.info(self,'Rmax for surface %f', self.rmax)

        logger.info(self,'* Radial and angular grid Info')
        logger.info(self,'Will be the grid pruned ? %s' % self.prune)
        logger.info(self,'Rho*W cutoff %e' % self.small_rho_cutoff)
        logger.info(self,'Npang points inside %d' % self.bnpang)
        logger.info(self,'Number of radial points outside %d', self.nrad)
        logger.info(self,'Number of radial points inside %d', self.bnrad)
        logger.info(self,'Radial outside quadrature %s', self.iqudr)
        logger.info(self,'Radial outside mapping %s', self.mapr)
        logger.info(self,'Radial inside quadrature %s', self.biqudr)
        logger.info(self,'Radial inside mapping %s', self.bmapr)
        logger.info(self,'Slater-Bragg radii %f', self.rad) 
        logger.info(self,'Beta-Sphere factor %f', self.betafac)
        logger.info(self,'Beta-Sphere radi %f', self.brad)
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
        if self.charges[self.inuc] == 1:
            self.rad = grid.BRAGG[self.charges[self.inuc]]
        else:
            self.rad = grid.BRAGG[self.charges[self.inuc]]*0.5

        idx = 'atom'+str(self.inuc)
        with h5py.File(self.surfile) as f:
            self.xnuc = f[idx+'/xnuc'].value
            self.xyzrho = f[idx+'/xyzrho'].value
            self.npang = f[idx+'/npang'].value
            self.ntrial = f[idx+'/ntrial'].value
            self.rmin = f[idx+'/rmin'].value
            self.rmax = f[idx+'/rmax'].value
            self.rsurf = f[idx+'/rsurf'].value
            self.nlimsurf = f[idx+'/nlimsurf'].value
            self.agrids = f[idx+'/coords'].value

        self.brad = self.rmin*self.betafac

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        if (self.iqudr == 'legendre'):
            self.iqudr = 1
        if (self.biqudr == 'legendre'):
            self.biqudr = 1

        if (self.mapr == 'becke'):
            self.mapr = 1
        elif (self.mapr == 'exp'):
            self.mapr = 2
        elif (self.mapr == 'none'):
            self.mapr = 0 
        if (self.bmapr == 'becke'):
            self.bmapr = 1
        elif (self.bmapr == 'exp'):
            self.bmapr = 2
        elif (self.bmapr == 'none'):
            self.bmapr = 0

        with lib.with_omp_threads(self.nthreads):
            bpoints, bweigths = int_beta(self)
            points, weigths = out_beta(self)

        self.w = numpy.hstack((bweigths, weigths))
        self.p = numpy.vstack((bpoints, points))
        logger.info(self,'Total number of points for atom %d %d' % (self.inuc,len(self.w)))
        if (self.prune):
            prune_small_rho_grids(self)
            logger.info(self,'Total number pruned points for atom %d %d' % (self.inuc,len(self.w)))

        logger.info(self,'Write info to HDF5 file')
        atom_dic = {'w':self.w,
                    'p':self.p}
        lib.chkfile.save(self.surfile, 'grid'+str(self.inuc), atom_dic)
        logger.info(self,'Points of atom %d done',self.inuc)
        logger.timer(self,'Points build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    natm = 3
    name = 'h2o.chk'
    bas = Points(name)
    bas.verbose = 4
    bas.nrad = 101
    bas.iqudr = 'legendre'
    bas.mapr = 'exp'
    bas.bnrad = 101
    bas.bnpang = 1202
    bas.biqudr = 'legendre'
    bas.bmapr = 'exp'
    bas.betafac = 0.4
    bas.prune = True
    for i in range(natm):
        bas.inuc = i
        bas.kernel()
        t0 = time.time()
        rhop = rho(bas,bas.p)
        rhov = numpy.dot(rhop,bas.w)
        logger.info(bas,'Atom %d density %f' % (i,rhov))
        logger.info(bas,'Time for density using precomputed grid %.3f (sec)' % (time.time()-t0))

