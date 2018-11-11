#!/usr/bin/env python

def ray(self,idx):

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

    ncount = 0
    nintersec = 0
    cost = self.grids[idx,0]
    sintcosp = self.grids[idx,1]*self.grids[idx,2]
    sintsinp = self.grids[idx,1]*self.grids[idx,3]
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
    self.nlimsurf[idx] = nintersec
    for ii in range(nintersec):
        self.rsurf[idx,ii] = xsurf[ii,2]
    if (nintersec%2 == 0):
        nintersec = +1
        self.nlimsurf[idx] += nintersec
        self.rsurf[idx,nintersec-1] = self.rmaxsurf
    print("#* ",idx,self.grids[idx,:4],self.rsurf[idx,:nintersec])
