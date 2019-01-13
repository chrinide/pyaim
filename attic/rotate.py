#!/usr/bin/env python
    def rotategrid(self):                    
    
        self.angx = self.angx*numpy.pi/180.0
        self.angy = self.angy*numpy.pi/180.0
        self.angz = self.angz*numpy.pi/180.0
        xmat = numpy.zeros((3,3))
        ymat = numpy.zeros((3,3))
        zmat = numpy.zeros((3,3))

        xmat[0,0] = 1.0
        cangx = numpy.cos(self.angx)
        sangx = numpy.sin(self.angx)
        xmat[1,1] = +cangx
        xmat[2,1] = +sangx
        xmat[1,2] = -sangx
        xmat[2,2] = +cangx

        ymat[1,1] = 1.0
        cangy = numpy.cos(self.angy)
        sangy = numpy.sin(self.angy)
        ymat[0,0] = +cangy
        ymat[2,0] = +sangy
        ymat[0,2] = -sangy
        ymat[2,2] = +cangy

        zmat[2,2] = 1.0
        cangz = numpy.cos(self.angz)
        sangz = numpy.sin(self.angz)
        zmat[0,0] = +cangz
        zmat[1,0] = +sangz
        zmat[0,1] = -sangz
        zmat[1,1] = +cangz

        # Full rotacion matrix. R = R_X * R_Y * R_Z
        xtmp = numpy.matmul(ymat,zmat)
        trot = numpy.matmul(xmat,xtmp)

        for i in range(self.npang): 
            x = self.grids[i,1]*self.grids[i,2]
            y = self.grids[i,1]*self.grids[i,3] 
            z = self.grids[i,0]
            xp = trot[0,0]*x + trot[0,1]*y + trot[0,2]*z
            yp = trot[1,0]*x + trot[1,1]*y + trot[1,2]*z
            zp = trot[2,0]*x + trot[2,1]*y + trot[2,2]*z
            rxy = xp*xp+yp*yp
            r = numpy.sqrt(rxy+zp*zp)
            if (rxy < EPS):
                if (zp >= 0.0):
                    self.grids[i,0] = +1.0
                else:
                    self.grids[i,0] = -1.0
                self.grids[i,1] = 0.0
                self.grids[i,3] = 0.0
                self.grids[i,2] = 1.0
            else:
              rxy = numpy.sqrt(rxy)
              self.grids[i,0] = zp/r                          
              self.grids[i,1] = numpy.sqrt((1.0-self.grids[i,0])*(1.0+self.grids[i,0])) 
              self.grids[i,2] = xp/rxy                        
              self.grids[i,3] = yp/rxy                        
