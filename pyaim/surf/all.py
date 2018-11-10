#!/usr/bin/env python

import os
import sys
import time
import numpy
import ctypes
import h5py
import gc

from pyscf import dft
from pyscf import lib
from pyscf.lib import logger
libdft = lib.load_library('libdft')

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

EPS = 1e-7
GRADEPS = 1e-10
RHOEPS = 1e-6
MINSTEP = 1e-6
MAXSTEP = 4.0
SAFETY = 0.9
HMINIMAL = numpy.finfo(numpy.float64).eps
LEBEDEV_NGRID = numpy.asarray((
    1   , 6   , 14  , 26  , 38  , 50  , 74  , 86  , 110 , 146 ,
    170 , 194 , 230 , 266 , 302 , 350 , 434 , 590 , 770 , 974 ,
    1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810))

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
        self.mol = None
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
        self._keys = set(self.__dict__.keys())

    def dump_input(self):
        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'Surface file %s' % self.surfile)
        logger.info(self,'max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        logger.info(self,'Surface for nuc %d' % self.inuc)
        logger.info(self,'Nuclear coordinate %8.5f %8.5f %8.5f', *self.xnuc)
        logger.info(self,'Rmaxsurface %8.5f' % self.rmaxsurf)
        logger.info(self,'Ntrial %d' % self.ntrial)
        logger.info(self,'Npang points %d' % self.npang)
        logger.info(self,'Rprimer %8.5f' % self.rprimer)
        logger.info(self,'Epsiscp %8.5f' % self.epsiscp)
        logger.info(self,'Epsroot %8.5f' % self.epsroot)
        logger.info(self,'ODE solver %s' % self.backend)
        logger.info(self,'ODE tool %8.5f' % self.epsilon)
        logger.info(self,'Max steps in ODE solver %d' % self.mstep)
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
        lib.logger.TIMER_LEVEL = 5

        self.mol = lib.chkfile.load_mol(self.chkfile)
        self.natm = self.mol.natm		
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in self.mol._atom])
        self.charges = self.mol.atom_charges()
        self.mo_coeff = lib.chkfile.load(self.chkfile, 'scf/mo_coeff')
        self.mo_occ = lib.chkfile.load(self.chkfile, 'scf/mo_occ')

        if (self.ntrial%2 == 0): self.ntrial += 1
        geofac = numpy.power(((self.rmaxsurf-0.1)/self.rprimer),(1.0/(self.ntrial-1.0)))
        self.rpru = numpy.zeros((self.ntrial))
        for i in range(self.ntrial): 
            self.rpru[i] = self.rprimer*numpy.power(geofac,(i+1)-1)
        self.lebgrid()
        self.xnuc = self.coords[self.inuc]
        self.rsurf = numpy.zeros((self.npang,self.ntrial))
        self.nlimsurf = numpy.zeros((self.npang), dtype=numpy.int32)
        if self.verbose > lib.logger.NOTE:
            self.dump_input()
        self.xyzrho, gradmod = self.gradrho(self.xnuc,self.step)
        if (gradmod > 1e-4):
            if (self.charges[self.inuc] > 2.0):
                logger.info(self,'Check rho position %8.5f %8.5f %8.5f', *self.xyzrho)
            else:
                raise RuntimeError('Failed finding nucleus:', *self.xyzrho) 
        else:
            logger.info(self,'Check rho position %8.5f %8.5f %8.5f', *self.xyzrho)

        self.npang = 40
        self.surface()

        self.rmin = 1000.0
        self.rmax = 0.0
        for i in range(self.npang):
            nsurf = int(self.nlimsurf[i])
            self.rmin = numpy.minimum(self.rmin,self.rsurf[i,0])
            self.rmax = numpy.maximum(self.rmax,self.rsurf[i,nsurf-1])
        logger.info(self,'Rmin for surface %8.5f', self.rmin)
        logger.info(self,'Rmax for surface %8.5f', self.rmax)
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
                ier, xpoint, rho, gradmod = self.odeint(xpoint,self.step)
                #rho, grad, gradmod = self.rhograd(xpoint)
                good, ib = self.checkcp(xpoint,rho,gradmod)
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
                    ier, xpoint, rho, gradmod = self.odeint(xpoint,self.step)
                    #rho, grad, gradmod = self.rhograd(xpoint)
                    good, im = self.checkcp(xpoint,rho,gradmod)
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

    # ier = 0 (correct), 1 (short step), 2 (too many iterations), 3 (infty)
    def odeint(self, xpoint, h):

        ier = 0
        h0 = h

        rho, grad, gradmod = self.rhograd(xpoint)
        if (gradmod <= GRADEPS and rho <= RHOEPS):
            ier = 3
            return ier, xpoint, rho, gradmod

        for nstep in range(self.mstep):
            xlast = xpoint
            ok, xpoint, h0 = self.adaptive_stepper(xpoint,grad,h0)
            if (ok == False):
                xpoint = xlast
                ier = 1
                #logger.info(self,'err 1: nsteps in odeint %d', nstep)
                return ier, xpoint, rho, gradmod
            rho, grad, gradmod = self.rhograd(xpoint)
            iscp, nuc = self.checkcp(xpoint,rho,gradmod)
            #logger.debug(self,'iscp %s xpoint %8.5f %8.5f %8.5f rho %8.5 gradmod %8.5f', \
            #             iscp, xpoint[0], xpoint[1], xpoint[2], rho, gradmod)
            if (iscp == True):
                ier = 0
                #logger.info(self,'err 0: nsteps in odeint %d', nstep)
                return ier, xpoint, rho, gradmod

        ier = 2
        logger.info(self,'maybe NNA at %8.5f %8.5f %8.f try to increase epsiscp', *xpoint)
        #logger.info(self,'err 2: nsteps in odeint %d', nstep)
        return ier, xpoint, rho, gradmod

    def gradrho(self, xpoint, h):

        h0 = h
        niter = 0
        rho, grad, gradmod = self.rhograd(xpoint)
        grdt = grad
        grdmodule = gradmod

        while (grdmodule > GRADEPS and niter < self.mstep):
            niter += 1
            ier = 1
            while (ier != 0):
                #xtemp, xerr = self.stepper_rkck(xpoint, grdt, h0)
                xtemp, xerr = self.stepper_he(xpoint, grdt, h0)
                rho, grad, gradmod = self.rhograd(xtemp)
                escalar = numpy.einsum('i,i->',grdt,grad) 
                if (escalar < 0.707):
                    if (h0 >= MINSTEP):
                        h0 = h0/2.0
                        ier = 1
                    else:
                        ier = 0
                else:
                    if (escalar > 0.9): 
                        hproo = h0*1.6
                        if (hproo < h):
                            h0 = hproo
                        else:
                            h0 = h
                    ier = 0
                    xpoint = xtemp
                    grdt = grad
                    grdmodule = gradmod
                #logger.debug(self,'angle, step in gradrho %8.5f %8.5f', escalar, h0)

        #logger.debug(self,'nsteps in gradrho %d', niter)
        return xpoint, grdmodule

    def rhograd(self, x):

        x = numpy.reshape(x, (-1,3))
        ao = dft.numint.eval_ao(self.mol, x, deriv=1)
        rho = dft.numint.eval_rho2(self.mol, ao, self.mo_coeff, self.mo_occ, xctype='GGA')
        gradmod = numpy.linalg.norm(rho[-3:,0])

        return rho[0,0], rho[-3:,0]/(gradmod+HMINIMAL), gradmod

    def checkcp(self, x, rho, gradmod):

        iscp = False
        nuc = -2

        for i in range(self.natm):
            r = numpy.linalg.norm(x-self.coords[i])
            if (r < self.epsiscp):
                iscp = True
                nuc = i
                return iscp, nuc

        if (gradmod <= GRADEPS):
            iscp = True
            if (rho <= RHOEPS): 
                nuc = -1

        return iscp, nuc

    def adaptive_stepper(self, x, grad, h):

        adaptive = True 
        ier = 1
        xtemp = numpy.zeros((3))
        xerrv = numpy.zeros((3))

        while (ier != 0):
            xtemp, xerrv = self.stepper_rkck(x, grad, h)
            nerr = numpy.linalg.norm(xerrv)
            if (nerr < self.epsilon):
                ier = 0
                x = xtemp
                if (nerr < self.epsilon/10.0): 
                    h = numpy.minimum(MAXSTEP, 1.6*h)
                #logger.debug(self,'point %8.5f %8.5f %8.5f error %8.5f step %8.5f', x[0],x[1],x[2],nerr,h)
            else:
                scale = SAFETY*(self.epsilon/nerr)
                h = scale*h
                #logger.debug(self,'point %8.5f %8.5f %8.5f error %8.5f step %8.5f', x[0],x[1],x[2],nerr,h)
                if (abs(h) < MINSTEP):
                    adaptive = False 
                    return adaptive, x, h 

        return adaptive, x, h

    # Heun-Euler
    def stepper_he(self, xpoint, grdt, h0):

        b21 = 1.0
        c1 = 1.0/2.0
        c2 = 1.0/2.0
        dc1 = c1 - 1.0
        dc2 = c2

        xout = xpoint + h0*b21*grdt
        rho, grad, gradmod = self.rhograd(xout)
        ak2 = grad

        xout = xpoint + h0*(c1*grdt+c2*ak2)
        xerr = h0*(dc1*grdt+dc2*ak2)

        return xout, xerr

    # Runge-Kutta-Cash-Karp
    def stepper_rkck(self, xpoint, grdt, h0):

        b21 = 1.0/5.0
        b31 = 3.0/40.0 
        b32 = 9.0/40.0
        b41 = 3.0/10.0 
        b42 = -9.0/10.0
        b43 = 6.0/5.0
        b51 = -11.0/54.0
        b52 = 5.0/2.0
        b53 = -70.0/27.0
        b54 = 35.0/27.0
        b61 = 1631.0/55296.0
        b62 = 175.0/512.0
        b63 = 575.0/13824.0
        b64 = 44275.0/110592.0
        b65 = 253.0/4096.0
        c1 = 37.0/378.0
        c3 = 250.0/621.0
        c4 = 125.0/594.0
        c6 = 512.0/1771.0
        dc1 = c1-(2825.0/27648.0)
        dc3 = c3-(18575.0/48384.0)
        dc4 = c4-(13525.0/55296.0)
        dc5 = -277.0/14336.0
        dc6 = c6-(1.0/4.0)
    
        xout = xpoint + h0*b21*grdt

        rho, grad, gradmod = self.rhograd(xout)
        ak2 = grad
        xout = xpoint + h0*(b31*grdt+b32*ak2)

        rho, grad, gradmod = self.rhograd(xout)
        ak3 = grad
        xout = xpoint + h0*(b41*grdt+b42*ak2+b43*ak3)

        rho, grad, gradmod = self.rhograd(xout)
        ak4 = grad
        xout = xpoint + h0*(b51*grdt+b52*ak2+b53*ak3+b54*ak4)

        rho, grad, gradmod = self.rhograd(xout)
        ak5 = grad
        xout = xpoint + h0*(b61*grdt+b62*ak2+b63*ak3+b64*ak4+b65*ak5)

        rho, grad, gradmod = self.rhograd(xout)
        ak6 = grad
        xout = xpoint + h0*(c1*grdt+c3*ak3+c4*ak4+c6*ak6)

        xerr = h0*(dc1*grdt+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)

        return xout, xerr

    def lebgrid(self):

        if self.npang not in LEBEDEV_NGRID:
            raise ValueError('Unsupported angular grids %d' % self.npang)
        else:
            grids = numpy.zeros((self.npang,4))
            self.grids = numpy.zeros((self.npang,5))
            libdft.MakeAngularGrid(grids.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(self.npang))

        for i in range(self.npang):
            self.grids[i,4] = 4.0*numpy.pi*grids[i,3]
            rxy = grids[i,0]*grids[i,0] + grids[i,1]*grids[i,1]
            r = numpy.sqrt(rxy + grids[i,2]*grids[i,2])
            if (rxy < EPS):
                if (grids[i,2] >= 0.0):
                    self.grids[i,0] = +1.0
                else:
                    self.grids[i,0] = -1.0
                self.grids[i,1] = 0.0
                self.grids[i,3] = 0.0
                self.grids[i,2] = 1.0
            else:
                rxy = numpy.sqrt(rxy)
                self.grids[i,0] = grids[i,2]/r
                self.grids[i,1] = numpy.sqrt((1.0-self.grids[i,0])*(1.0+self.grids[i,0]))
                self.grids[i,2] = grids[i,0]/rxy
                self.grids[i,3] = grids[i,1]/rxy

if __name__ == '__main__':
    name = 'h2o.chk'
    #name = 'c2f4.chk'
    surf = BaderSurf(name)
    surf.verbose = 5
    surf.inuc = 0
    surf.kernel()

