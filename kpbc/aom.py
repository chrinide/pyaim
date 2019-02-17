#!/usr/bin/env python

import sys
import time
import h5py
import numpy
import signal

from pyscf import lib
from pyscf.pbc import dft
from pyscf.lib import logger
from pyscf.pbc import lib as libpbc
from pyscf.tools.dump_mat import dump_tri

import grid

signal.signal(signal.SIGINT, signal.SIG_DFL)

EPS = 1e-7
NCOL = 15
DIGITS = 5

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

# TODO: screaning of points
# Temporal only occupied orbitals
def mos(self,x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao_kpts(self.cell, x, kpts=self.kpts, deriv=0)
    nocc = self.nmo
    npoints, nao = ao[0].shape
    aom = numpy.zeros((nocc*(nocc+1)/2,npoints), dtype=numpy.complex128)
    idx = 0
    for i in range(nocc):
        cposi = self.mo_coeff[:,i]
        ii = self.tags[i]
        c0i = numpy.dot(ao[ii], cposi)
        for j in range(i+1):
            cposj = self.mo_coeff[:,j]
            jj = self.tags[j]
            c0j = numpy.dot(ao[jj], cposj)
            aom[idx] = numpy.einsum('i,i->i',c0i.conj(),c0j)
            idx += 1
    return aom

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
    nocc = self.nmo
    NPROPS = nocc*(nocc+1)//2
    rprops = numpy.zeros(NPROPS, dtype=numpy.complex128)
    for n in range(nrad):
        r = rmesh[n]
        coords = []
        weigths = []
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
                weigths.append(coordsang[j,4])
        coords = numpy.array(coords)
        weigths = numpy.array(weigths)
        val = mos(self,coords)
        props = numpy.einsum('pi,i->p', val, weigths)
        rprops += props*dvol[n]*rwei[n]
    logger.info(self,'Time out Bsphere %.3f (sec)' % (time.time()-t0))
    return rprops
    
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
    nocc = self.nmo
    NPROPS = nocc*(nocc+1)//2
    rprops = numpy.zeros(NPROPS, dtype=numpy.complex128)
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
            coords[j] = p
        val = mos(self,coords)
        props = numpy.einsum('pi,i->p', val, coordsang[:,4])
        rprops += props*dvol[n]*rwei[n]
    logger.info(self,'Time in Bsphere %.3f (sec)' % (time.time()-t0))
    return rprops

class Aom(lib.StreamObject):

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
        self.non0tab = False
        self.orbs = -1
##################################################
# don't modify the following attributes, they are not input options
        self.a = None
        self.b = None
        self.vol = None
        self.cell = None
        self.kpts = None
        self.nkpts = None
        self.tags = None
        self.ls = None
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
        self.nprims = None
        self.nmo = None
        self.rad = None
        self.brad = None
        self.aom = None
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

        logger.info(self,'* Cell Info')
        logger.info(self,'Cell dimension : %d', self.cell.dimension)
        logger.info(self,'Lattice vectors (Bohr)')
        for i in range(3):
            logger.info(self,'Cell a%d axis : %.6f  %.6f  %.6f', i, *self.a[i])
        logger.info(self,'Lattice reciprocal vectors (1/Bohr)')
        for i in range(3):
            logger.info(self,'Cell b%d axis : %.6f  %.6f  %.6f', i, *self.b[i])
        logger.info(self,'Cell volume %g (Bohr^3)', self.vol)
        logger.info(self,'Number of cell vectors %d' % len(self.ls))
        logger.info(self,'Number of kpoints %d ' % self.nkpts)
        for i in range(self.nkpts):
            logger.info(self,'K-point %d : %.6f  %.6f  %.6f', 1, *self.kpts[i])
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])

        logger.info(self,'* Basis Info')
        logger.info(self,'Number of molecular orbitals %d' % self.nmo)
        logger.info(self,'Number of molecular primitives %d' % self.nprims)

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

        logger.info(self,'* AOM Info')
        logger.info(self,'List of orbitals per K point to be integrated %s', self.orbs)
        logger.info(self,'')

        return self

    def build(self):

        t0 = (time.clock(), time.time())
        lib.logger.TIMER_LEVEL = 3

        self.cell = libpbc.chkfile.load_cell(self.chkfile)
        self.a = self.cell.lattice_vectors()
        self.b = self.cell.reciprocal_vectors()
        self.vol = self.cell.vol
        self.nelectron = self.cell.nelectron 
        self.charge = self.cell.charge    
        self.spin = self.cell.spin      
        self.natm = self.cell.natm		
        self.ls = self.cell.get_lattice_Ls(dimension=self.cell.dimension)
        self.ls = self.ls[numpy.argsort(lib.norm(self.ls, axis=1))]
        self.kpts = lib.chkfile.load(self.chkfile, 'kcell/kpts')
        self.nkpts = len(self.kpts)
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.cell._atom])
        self.charges = self.cell.atom_charges()
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

        # TODO: general orbitals
        mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        nprims, nmo = mo_coeff[0].shape 
        if (self.orbs == -1):
            self.orbs = nmo
        self.nprims = nprims
        self.nmo = self.orbs*self.nkpts
        self.mo_coeff = numpy.zeros((self.nprims,self.nmo), dtype=numpy.complex128)
        self.tags = numpy.zeros(self.nmo, dtype=numpy.int32)
        ii = 0
        for k in range(self.nkpts):
            for i in range(self.orbs):
                self.mo_coeff[:,ii] = mo_coeff[k][:,i]
                self.tags[ii] = k
                ii += 1

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

        self.aom = numpy.zeros((self.nmo,self.nmo), dtype=numpy.complex128)
        with lib.with_omp_threads(self.nthreads):
            aomb = int_beta(self)
            aoma = out_beta(self)
        
        idx = 0
        for i in range(self.nmo):
            for j in range(i+1):
                self.aom[i,j] = aoma[idx]+aomb[idx] 
                self.aom[j,i] = self.aom[i,j].conj()
                idx += 1
        norm = float(1.0/self.nkpts)
        self.aom = self.aom*norm
        if (self.nmo<=30):
            dump_tri(self.stdout, self.aom, ncol=NCOL, digits=DIGITS, start=0)

        logger.info(self,'Write info to HDF5 file')
        atom_dic = {'aom':self.aom}
        lib.chkfile.save(self.surfile, 'ovlp'+str(self.inuc), atom_dic)
        logger.info(self,'')

        logger.info(self,'AOM of atom %d done',self.inuc)
        logger.timer(self,'AOM build', *t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'kpts.chk'
    ovlp = Aom(name)
    ovlp.verbose = 4
    ovlp.nrad = 221
    ovlp.iqudr = 'legendre'
    ovlp.mapr = 'none'
    ovlp.bnrad = 121
    ovlp.bnpang = 3074
    ovlp.biqudr = 'legendre'
    ovlp.bmapr = 'exp'
    ovlp.betafac = 0.4
    ovlp.inuc = 0
    ovlp.kernel()

