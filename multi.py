#!/usr/bin/env python

import os
import sys
import time
import numpy
import h5py
import ctypes
import signal
from pyscf import lib
from pyscf.lib import logger

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)
libcgto = lib.load_library('libcgto')

name = 'h2o.chk.h5'

idx = 1
jdx = 2
        
idx1 = 'qlm'+str(idx)
idx2 = 'qlm'+str(jdx)
with h5py.File(name) as f:
    lmax1 = f[idx1+'/lmax'].value
    qlm1 = f[idx1+'/totprops'].value
    lmax2 = f[idx2+'/lmax'].value
    qlm2 = f[idx2+'/totprops'].value
idx1 = 'atom'+str(idx)
with h5py.File(name) as f:
    xyzrho = f[idx1+'/xyzrho'].value
    
rint = numpy.zeros(3)
rint = xyzrho[jdx] - xyzrho[idx]

ncent = xyzrho.shape[0]
lmax = lmax1
NPROPS = lmax*(lmax+2) + 1
print "Multipolar interaction for atoms", idx, jdx
print "Lmax", lmax

coeff = numpy.zeros((NPROPS,NPROPS))
coulm = numpy.zeros((NPROPS,NPROPS))
jlm = numpy.zeros((NPROPS),dtype=numpy.int32)
coulp = numpy.zeros(lmax+1)

feval = 'eval_gaunt'
drv = getattr(libaim, feval)
drv(ctypes.c_int(lmax), 
    rint.ctypes.data_as(ctypes.c_void_p),
    coeff.ctypes.data_as(ctypes.c_void_p), 
    jlm.ctypes.data_as(ctypes.c_void_p))

coul = 0.0
for lm1 in range(0,NPROPS):
    for lm2 in range(0,NPROPS):
        q1 = qlm1[lm1]
        q2 = qlm2[lm2]
        coulm[lm1,lm2] = q1*q2*coeff[lm2,lm1]
        coul += coulm[lm1,lm2]
print "Total Multipolar Coulomb interaction", coul

for lm1 in range(0,NPROPS):
    l1 = jlm[lm1]
    for lm2 in range(0,NPROPS):
        l2 = jlm[lm2]
        l = max(l1,l2)
        coulp[l] += coulm[lm1,lm2]
for l in range(1,lmax+1):
    coulp[l] = coulp[l-1]+coulp[l]

print "Multipolar Coulomb interaction in l"
for l in range(0,lmax+1):
    print "l,coulp",l,coulp[l]

