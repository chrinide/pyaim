#!/usr/bin/env python

import os
import sys
import time
import numpy
import ctypes
import h5py

from pyscf import lib
from pyscf.lib import logger
libcgto = lib.load_library('libcgto')
lib.num_threads(1)

from pyaim.surf import ode, cp, grid

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

def surface(self):

    xin = numpy.zeros((3))
    xfin = numpy.zeros((3))
    xmed = numpy.zeros((3))
    xpoint = numpy.zeros((3))
    xdeltain = numpy.zeros((3))
    xsurf = numpy.zeros((self.ntrial,3))
    isurf = numpy.zeros((self.ntrial,2), dtype=numpy.int32)

    if (self.natm == 1):
        self.nlimsurf[:] = 1
        self.rsurf[:,0] = self.rmaxsurf
        return

    for i in range(self.npang):
        ncount = 0
        nintersec = 0
        cost = self.grids[i,0]
        sintcosp = self.grids[i,1]*self.grids[i,2]
        sintsinp = self.grids[i,1]*self.grids[i,3]
        ia = self.inuc
        ra = 0.0
        for j in range(self.ntrial):
            ract = self.rpru[j]
            xdeltain[0] = ract*sintcosp
            xdeltain[1] = ract*sintsinp
            xdeltain[2] = ract*cost    
            xpoint = self.xnuc + xdeltain
            ier, xpoint, rho, gradmod = ode.odeint(self,xpoint)
            good, ib = cp.checkcp(self,xpoint,rho,gradmod)
            rb = ract
            if (ib != ia and (ia == self.inuc or ib == self.inuc)):
                if (ia != self.inuc or ib != -1):
                    nintersec += 1
                    xsurf[nintersec-1,0] = ra
                    xsurf[nintersec-1,1] = rb
                    isurf[nintersec-1,0] = ia
                    isurf[nintersec-1,1] = ib
            ia = ib
            ra = rb
        for k in range(nintersec):
            ia = isurf[k,0]
            ib = isurf[k,1]
            ra = xsurf[k,0]
            rb = xsurf[k,1]
            xin[0] = self.xnuc[0] + ra*sintcosp
            xin[1] = self.xnuc[1] + ra*sintsinp
            xin[2] = self.xnuc[2] + ra*cost
            xfin[0] = self.xnuc[0] + rb*sintcosp
            xfin[1] = self.xnuc[1] + rb*sintsinp
            xfin[2] = self.xnuc[2] + rb*cost
            while (abs(ra-rb) > self.epsroot):
                xmed = 0.5*(xfin+xin)    
                rm = 0.5*(ra+rb)
                xpoint = xmed
                ier, xpoint, rho, gradmod = ode.odeint(self,xpoint)
                good, im = cp.checkcp(self,xpoint,rho,gradmod)
                #if (ib != -1 and (im != ia and im != ib)):
                    #logger.debug(self,'warning new intersections found')
                if (im == ia):
                    xin = xmed
                    ra = rm
                elif (im == ib):
                    xfin = xmed
                    rb = rm
                else:
                    if (ia == self.inuc):
                        xfin = xmed
                        rb = rm
                    else:
                        xin = xmed
                        ra = rm
            xpoint = 0.5*(xfin+xin)    
            xsurf[k,2] = 0.5*(ra+rb)
        
        # organize pairs
        self.nlimsurf[i] = nintersec
        for ii in range(nintersec):
            self.rsurf[i,ii] = xsurf[ii,2]
        if (nintersec%2 == 0):
            nintersec = +1
            self.nlimsurf[i] += nintersec
            self.rsurf[i,nintersec-1] = self.rmaxsurf
        print("#* ",i,self.grids[i,:4],self.rsurf[i,:nintersec])

class BaderSurf(lib.StreamObject):

    def __init__(self, datafile):
        self.inuc = 0
        self.epsiscp = 0.180
        self.ntrial = 11
        self.npang = 5810
        self.epsroot = 1e-4
        self.rmaxsurf = 10.0
        self.rprimer = 0.4
        self.backend = 'rkck'
        self.epsilon = 1e-4 
        self.step = 0.1
        self.mstep = 100
        self.surfile = 'surface.h5'
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.scratch = lib.param.TMPDIR 
##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = None
        self.mo_occ = None
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
        self.env = None
        self.ao_loc = None
        self.shls_slice = None
        self.nao = None
        self.non0tab = None
        self.cart = None
        self._keys = set(self.__dict__.keys())

    def dump_input(self):

        if self.verbose < logger.INFO:
            return self

        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'Surface for nuc %d' % self.inuc)
        logger.info(self,'Nuclear coordinate %.6f  %.6f  %.6f', *self.xnuc)
        logger.info(self,'Rmaxsurface %.6f' % self.rmaxsurf)
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Rprimer %.6f' % self.rprimer)
        logger.debug(self, 'Rpru : %s' % self.rpru) 
        logger.info(self,'Epsiscp %.6f' % self.epsiscp)
        logger.info(self,'Epsroot %.6f' % self.epsroot)
        logger.info(self,'ODE solver %s' % self.backend)
        logger.info(self,'ODE tool %.6f' % self.epsilon)
        logger.info(self,'Max steps in ODE solver %d' % self.mstep)

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 5

        mol = lib.chkfile.load_mol(self.chkfile)
        self.nelectron = mol.nelectron 
        self.charge = mol.charge    
        self.spin = mol.spin      
        self.natm = mol.natm		
        self.atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
        self.bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
        self.nbas = self.bas.shape[0]
        self.env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
        self.ao_loc = mol.ao_loc_nr()
        self.shls_slice = (0, self.nbas)
        sh0, sh1 = self.shls_slice
        self.nao = self.ao_loc[sh1] - self.ao_loc[sh0]
        self.non0tab = numpy.ones((1,self.nbas), dtype=numpy.int8)
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in mol._atom])
        self.charges = mol.atom_charges()
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')
        self.cart = mol.cart

        if (self.ntrial%2 == 0): self.ntrial += 1
        geofac = numpy.power(((self.rmaxsurf-0.1)/self.rprimer),(1.0/(self.ntrial-1.0)))
        self.rpru = numpy.zeros((self.ntrial))
        for i in range(self.ntrial): 
            self.rpru[i] = self.rprimer*numpy.power(geofac,(i+1)-1)
        self.xnuc = self.coords[self.inuc]
        self.rsurf = numpy.zeros((self.npang,self.ntrial))
        self.nlimsurf = numpy.zeros((self.npang), dtype=numpy.int32)
        self.grids = grid.lebgrid(self.npang)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > lib.logger.NOTE:
            self.dump_input()

        self.xyzrho, gradmod = cp.gradrho(self,self.xnuc,self.step)
        if (gradmod > 1e-4):
            if (self.charges[self.inuc] > 2.0):
                logger.info(self,'Check rho position %.6f %.6f %.6f', *self.xyzrho)
            else:
                raise RuntimeError('Failed finding nucleus:', *self.xyzrho) 
        else:
            logger.info(self,'Check rho position %.6f %.6f %.6f', *self.xyzrho)

        surface(self)

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
                    'intersecs':self.nlimsurf,
                    'surface':self.rsurf,
                    'npang':self.npang,
                    'rmin':self.rmin,
                    'rmax':self.rmax,
                    'ntrial':self.ntrial}
        logger.info(self,'Surface of atom %d saved',self.inuc)
        lib.chkfile.save(self.surfile, 'atom'+str(self.inuc), atom_dic)

        logger.timer(self,'BaderSurf build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'test/n2_rhf.chk'
    surf = BaderSurf(name)
    surf.epsilon = 1e-4
    surf.verbose = 4
    surf.inuc = 0
    surf.kernel()
 
