#!/usr/bin/env python

import ctypes
import numpy
from pyscf import lib

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c
libcgto = lib.load_library('libcgto')
OCCDROP = 1e-12

def eval_rho3(mol, coords, mo_coeff, mo_occ, deriv=0):

    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if mol.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv

    # Toda esta info puede ponerse en memoria global al inicio del programa
    # y reusar para evitar crear en cada llamda los punteros
    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    coords = numpy.asarray(coords, dtype=numpy.double, order='F')
    natm = atm.shape[0]
    nbas = bas.shape[0]
    ngrids = coords.shape[0]
    ao_loc = mol.ao_loc_nr()
    shls_slice = (0, nbas)
    sh0, sh1 = shls_slice
    nao = ao_loc[sh1] - ao_loc[sh0]
    ao = numpy.ndarray((comp,nao,ngrids))
    non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas), dtype=numpy.int8)

    drv = getattr(libcgto, feval)
    drv(ctypes.c_int(ngrids),
        (ctypes.c_int*2)(*shls_slice), ao_loc.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p))

    ao = numpy.swapaxes(ao, -1, -2)
    if comp == 1:
        ao = ao[0]

    pos = mo_occ > OCCDROP
    if pos.sum() > 0:
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
        if (deriv == 0):
            c0 = lib.dot(ao, cpos)
            rho = numpy.einsum('pi,pi->p', c0, c0)
        elif (deriv == 1):
            rho = numpy.empty((4,ngrids))
            c0 = lib.dot(ao[0], cpos)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            c1 = lib.dot(ao[1], cpos)
            rho[1] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
            c1 = lib.dot(ao[2], cpos)
            rho[2] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
            c1 = lib.dot(ao[3], cpos)
            rho[3] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
        else: 
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((12,ngrids))
            c0 = lib.dot(ao[0], cpos)
            rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[5] = 0

            c1x = lib.dot(ao[1], cpos)
            rho[5] += numpy.einsum('pi,pi->p', c1x, c1x)
            rho[1] = numpy.einsum('pi,pi->p', c0, c1x) * 2 # *2 for +c.c.

            c1y = lib.dot(ao[2], cpos)
            rho[5] += numpy.einsum('pi,pi->p', c1y, c1y)
            rho[2] = numpy.einsum('pi,pi->p', c0, c1y) * 2 # *2 for +c.c.

            c1z = lib.dot(ao[3], cpos)
            rho[5] += numpy.einsum('pi,pi->p', c1z, c1z)
            rho[3] = numpy.einsum('pi,pi->p', c0, c1z) * 2 # *2 for +c.c.

            XX, YY, ZZ = 4, 7, 9
            XY, XZ, YZ = 5, 6, 8
            
            ao2 = ao[XX]
            c1 = lib.dot(ao2, cpos)
            hessxx = numpy.einsum('pi,pi->p', c0, c1)
            hessxx += numpy.einsum('pi,pi->p', c1x, c1x) 
            hessxx *= 2.0
            ao2 = ao[YY]
            c1 = lib.dot(ao2, cpos)
            hessyy = numpy.einsum('pi,pi->p', c0, c1)
            hessyy += numpy.einsum('pi,pi->p', c1y, c1y) 
            hessyy *= 2.0
            ao2 = ao[ZZ]
            c1 = lib.dot(ao2, cpos)
            hesszz = numpy.einsum('pi,pi->p', c0, c1)
            hesszz += numpy.einsum('pi,pi->p', c1z, c1z) 
            hesszz *= 2.0

            ao2 = ao[XY]
            c1 = lib.dot(ao2, cpos)
            hessxy = numpy.einsum('pi,pi->p', c0, c1)
            hessxy += numpy.einsum('pi,pi->p', c1x, c1y) 
            hessxy *= 2.0
            ao2 = ao[XZ]
            c1 = lib.dot(ao2, cpos)
            hessxz = numpy.einsum('pi,pi->p', c0, c1)
            hessxz += numpy.einsum('pi,pi->p', c1x, c1z) 
            hessxz *= 2.0
            ao2 = ao[YZ]
            c1 = lib.dot(ao2, cpos)
            hessyz = numpy.einsum('pi,pi->p', c0, c1)
            hessyz += numpy.einsum('pi,pi->p', c1y, c1z) 
            hessyz *= 2.0

            rho[4] = hessxx + hessyy + hesszz
            rho[5] *= 0.5

            rho[6] = hessxx
            rho[7] = hessxy
            rho[8] = hessxz
            rho[9] = hessyy
            rho[10] = hessyz
            rho[11] = hesszz

    else:
        if (deriv == 0):
            rho = numpy.zeros(ngrids)
        elif (deriv == 1):
            rho = numpy.zeros((4,ngrids))
        else:
            rho = numpy.zeros((12,ngrids))

    return rho

