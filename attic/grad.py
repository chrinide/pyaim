#!/usr/bin/env python

import numpy, sys, time, ctypes
from pyscf import lib, dft
from pyscf.lib import logger

BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c
libcgto = lib.load_library('libcgto')
OCCDROP = 1e-12

name = 'h2o'
mol = lib.chkfile.load_mol(name+'.chk')
mf = lib.chkfile.load(name+'.chk', 'scf')
mo_coeff = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ = lib.chkfile.load(name+'.chk', 'scf/mo_occ')

atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
natm = atm.shape[0]
nbas = bas.shape[0]
ao_loc = mol.ao_loc_nr()
shls_slice = (0, nbas)
sh0, sh1 = shls_slice
nao = ao_loc[sh1] - ao_loc[sh0]
ngrids = 1
non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,nbas), dtype=numpy.int8)

def eval_grad(coords, cart=False):

    deriv = 1
    if cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
            
    coords = numpy.asarray(coords, dtype=numpy.double, order='F')
    ao = numpy.zeros((4,nao,ngrids), dtype=numpy.double)

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
    pos = mo_occ > OCCDROP
    cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
    rho = numpy.empty((4,ngrids))
    c0 = numpy.dot(ao[0], cpos)
    rho[0] = numpy.einsum('pi,pi->p', c0, c0)
    c1 = numpy.dot(ao[1], cpos)
    rho[1] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
    c1 = numpy.dot(ao[2], cpos)
    rho[2] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
    c1 = numpy.dot(ao[3], cpos)
    rho[3] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.

    return rho

log = lib.logger.Logger(sys.stdout, 4)
log.verbose = 5
lib.logger.TIMER_LEVEL = 5

a = [0.0, 0.1, 1.0]
a = numpy.asarray(a)
a = numpy.reshape(a, (-1,3))
npoints = 100000
    
t0 = time.clock()
for i in range(npoints):
    rho = eval_grad(a)
log.timer('own', t0)

t0 = time.clock()
for i in range(npoints):
    ao = dft.numint.eval_ao(mol, a, deriv=1)
    rho = dft.numint.eval_rho2(mol, ao, mo_coeff, mo_occ, xctype='GGA')
log.timer('pyscf one by one', t0)

a = []
for i in range(npoints):
    a.append([0.0, 0.1, 1.0])
a = numpy.asarray(a)
t0 = time.clock()
ao = dft.numint.eval_ao(mol, a, deriv=1)
rho = dft.numint.eval_rho2(mol, ao, mo_coeff, mo_occ, xctype='GGA')
log.timer('pyscf all', t0)
    
