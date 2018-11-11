#!/usr/bin/env python

import numpy
from pyscf.lib import logger
from pyaim.surf import field, cp

GRADEPS = 1e-10
RHOEPS = 1e-10
MINSTEP = 1e-4
MAXSTEP = 0.75
SAFETY = 0.8
ENLARGE = 1.2
HMINIMAL = numpy.finfo(numpy.float64).eps

# ier = 0 (correct), 1 (short step), 2 (too many iterations), 3 (infty)
def odeint(self, xpoint):

    ier = 0
    h0 = self.step

    rho, grad, gradmod = field.rhograd(self,xpoint)
    if (gradmod <= GRADEPS and rho <= RHOEPS):
        ier = 3
        return ier, xpoint, rho, gradmod

    for nstep in range(self.mstep):
        xlast = xpoint
        ok, xpoint, h0 = adaptive_stepper(self,xpoint,grad,h0)
        if (ok == False):
            xpoint = xlast
            ier = 1
            #logger.debug(self,'err 1: nsteps in odeint %d', nstep)
            return ier, xpoint, rho, gradmod
        rho, grad, gradmod = field.rhograd(self,xpoint)
        iscp, nuc = cp.checkcp(self,xpoint,rho,gradmod)
        #logger.debug(self,'iscp %s xpoint %.6f %.6f %.6f rho %.6 gradmod %.6f' % \
        #             (iscp, xpoint[0], xpoint[1], xpoint[2], rho, gradmod))
        if (iscp == True):
            ier = 0
            #logger.debug(self,'err 0: nsteps in odeint %d', nstep)
            return ier, xpoint, rho, gradmod

    dist = numpy.linalg.norm(xpoint-self.xnuc)
    if (dist < self.rmaxsurf):
        logger.info(self,'maybe NNA at %.6f %.6f %.6f try to increase epsiscp', *xpoint)
    elif (dist >= self.rmaxsurf):
        ier = 3
        return ier, xpoint, rho, gradmod
    else:
        ier = 2
        #logger.debug(self,'err 2: nsteps in odeint %d', nstep)

    return ier, xpoint, rho, gradmod

def adaptive_stepper(self, x, grad, h):

    ier = 1
    adaptive = True 

    while (ier != 0):
        xtemp, xerrv = stepper_rkck(self, x, grad, h)
        nerr = numpy.linalg.norm(xerrv)/3.0
        if (nerr < self.epsilon):
            ier = 0
            x = xtemp
            if (nerr < self.epsilon/10.0): 
                h = numpy.minimum(MAXSTEP, ENLARGE*h)
            #logger.debug(self,'a- point %8.5f %8.5f %8.5f error %8.5f step %8.5f', x[0],x[1],x[2],nerr,h)
        else:
            scale = SAFETY*(self.epsilon/nerr)
            h = scale*h
            #logger.debug(self,'r- point %8.5f %8.5f %8.5f error %8.5f step %8.5f', x[0],x[1],x[2],nerr,h)
            if (abs(h) < MINSTEP):
                adaptive = False 
                return adaptive, x, h 

    return adaptive, x, h

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

    rho, grad, gradmod = field.rhograd(self,xout)
    ak2 = grad
    xout = xpoint + h0*(b31*grdt+b32*ak2)

    rho, grad, gradmod = field.rhograd(self,xout)
    ak3 = grad
    xout = xpoint + h0*(b41*grdt+b42*ak2+b43*ak3)

    rho, grad, gradmod = field.rhograd(self,xout)
    ak4 = grad
    xout = xpoint + h0*(b51*grdt+b52*ak2+b53*ak3+b54*ak4)

    rho, grad, gradmod = field.rhograd(self,xout)
    ak5 = grad
    xout = xpoint + h0*(b61*grdt+b62*ak2+b63*ak3+b64*ak4+b65*ak5)

    rho, grad, gradmod = field.rhograd(self,xout)
    ak6 = grad
    xout = xpoint + h0*(c1*grdt+c3*ak3+c4*ak4+c6*ak6)

    xerr = h0*(dc1*grdt+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)

    return xout, xerr
    
if __name__ == '__main__':
    from pyaim import surf
    name = 'test/lif.chk'
    surf = surf.BaderSurf(name)
    surf.npang = 1#770
    surf.build()
    #h0 = 0.1
    #xpoint = [0.55133747, 0.,        -0.73883567]
    xpoint = [-0.740658,0.000000,0.001822]
    #rho, grad, gradmod = field.rhograd(surf,xpoint)
    #print stepper_rkck(surf, xpoint, grad, h0) 
    #print adaptive_stepper(surf, xpoint, grad, h0)
    ier, xpoint, rho, gradmod = odeint(surf,xpoint)
    print ier, xpoint, rho, gradmod
    #ier, xpoint, rho, gradmod = odeint(surf,xpoint)
    #print ier, xpoint, rho, gradmod

