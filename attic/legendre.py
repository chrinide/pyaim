#! /usr/bin/env python

import numpy

def legendre(n):

    x = numpy.zeros(n)
    w = numpy.zeros(n)

    e1 = n*(n+1)
    m = ((n+1)//2)
    for i in range(1,m+1):
        mp1mi = m+1-i
        t = float(4*i-1)*numpy.pi/float(4*n+2)
        x0 = numpy.cos(t)*(1.0-(1.0-1.0/float(n))/float(8*n*n))
        pkm1 = 1.0
        pk = x0
        for k in range(2,n+1):
            pkp1 = 2.0*x0*pk-pkm1-(x0*pk-pkm1)/float(k)
            pkm1 = pk
            pk = pkp1
        d1 = float(n)*(pkm1-x0*pk)
        dpn = d1/(1.0-x0*x0)
        d2pn = (2.0*x0*dpn-e1*pk)/(1.0-x0*x0)
        d3pn = (4.0*x0*d2pn+(2.0-e1)*dpn)/(1.0-x0*x0)
        d4pn = (6.0*x0*d3pn+(6.0-e1)*d2pn)/(1.0-x0*x0)
        u = pk/dpn
        v = d2pn/dpn
        h = -u*(1.0+0.5*u*(v+u*(v*v-d3pn/(3.0*dpn))))
        p = pk+h*(dpn+0.5*h*(d2pn+h/3.0*(d3pn+0.25*h*d4pn)))
        dp = dpn+h*(d2pn+0.5*h*(d3pn+h*d4pn/3.0))
        h = h-p/dp
        xtemp = x0+h
        x[mp1mi-1] = xtemp
        fx = d1-h*e1*(pk+0.5*h*(dpn+h/3.0*(d2pn+0.25*h*(d3pn+0.2*h*d4pn))))
        w[mp1mi-1] = 2.0*(1.0-xtemp*xtemp)/fx/fx

    if ((n%2) == 1):
        x[0] = 0.0
    nmove = ((n+1)//2)
    ncopy = n-nmove
    for i in range(1,nmove+1):
        iback = n+1-i
        x[iback-1] = x[iback-ncopy-1]
        w[iback-1] = w[iback-ncopy-1]
    for i in range(1,n-nmove+1):
        x[i-1] = -x[n-i]
        w[i-1] = w[n-i]

    return x, w

if __name__ == '__main__':
    n = 10
    x, w = legendre(n) 
    print x
    print w

