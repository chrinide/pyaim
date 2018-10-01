#!/usr/bin/env python
    
import numpy, ctypes
from pyscf import lib

libcgto = lib.load_library('libcgto')

OCCDROP = 1e-12
HMINIMAL = numpy.finfo(numpy.float64).eps

def rhograd(self, x):

    deriv = 1
    if self.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    drv = getattr(libcgto, feval)

    x = numpy.reshape(x, (-1,3))
    x = numpy.asarray(x, dtype=numpy.double, order='F')
    ao = numpy.zeros((4,self.nao,1), dtype=numpy.double)

    drv(ctypes.c_int(1),
    (ctypes.c_int*2)(*self.shls_slice), 
    self.ao_loc.ctypes.data_as(ctypes.c_void_p),
    ao.ctypes.data_as(ctypes.c_void_p),
    x.ctypes.data_as(ctypes.c_void_p),
    self.non0tab.ctypes.data_as(ctypes.c_void_p),
    self.atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.natm),
    self.bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.nbas),
    self.env.ctypes.data_as(ctypes.c_void_p))

    ao = numpy.swapaxes(ao, -1, -2)
    pos = self.mo_occ > OCCDROP

    cpos = numpy.einsum('ij,j->ij', self.mo_coeff[:,pos], numpy.sqrt(self.mo_occ[pos]))
    rho = numpy.zeros((4,1))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    c1 = numpy.dot(ao[1], cpos)
    rho[1] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    c1 = numpy.dot(ao[2], cpos)
    rho[2] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    c1 = numpy.dot(ao[3], cpos)
    rho[3] = numpy.einsum('pi,pi->p', c0, c1)*2 # *2 for +c.c.
    gradmod = numpy.linalg.norm(rho[-3:,0])

    return rho[0,0], rho[-3:,0]/(gradmod+HMINIMAL), gradmod

#if __name__ == '__main__':

