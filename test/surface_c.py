#!/usr/bin/env python

import numpy, sys, time, ctypes
from pyscf import lib, dft
from pyscf.lib import logger
from pyaim.lib import utils

EPS = 1e-7
BLKSIZE = 128 # needs to be the same to lib/gto/grid_ao_drv.c
libcgto = lib.load_library('libcgto')
libdft = lib.load_library('libdft')
libsurf = lib.load_library('libsurf')
OCCDROP = 1e-12

name = 'h2o'
mol = lib.chkfile.load_mol(name+'.chk')
if mol.cart:
    cart = 1
else:
    cart = 0
mf = lib.chkfile.load(name+'.chk', 'scf')
mo_coeff = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ = lib.chkfile.load(name+'.chk', 'scf/mo_occ')

coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in mol._atom])
charges = mol.atom_charges()
atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
natm = atm.shape[0]
nbas = bas.shape[0]
nprim = mo_coeff.shape[0]
ao_loc = mol.ao_loc_nr()

log = lib.logger.Logger(sys.stdout, 4)
lib.logger.TIMER_LEVEL = 5

inuc = 0
epsiscp = 0.180
ntrial = 11
npang = 5810
epsroot = 1e-4
rmaxsurf = 10.0
rprimer = 0.4
backend = 1
epsilon = 1e-4 
step = 0.1
mstep = 100
if (ntrial%2 == 0): ntrial += 1
geofac = numpy.power(((rmaxsurf-0.1)/rprimer),(1.0/(ntrial-1.0)))
rpru = numpy.zeros((ntrial))
for i in range(ntrial): 
    rpru[i] = rprimer*numpy.power(geofac,(i+1)-1)

grids = numpy.zeros((npang,4))
grid = numpy.zeros((npang,5))
libdft.MakeAngularGrid(grids.ctypes.data_as(ctypes.c_void_p),ctypes.c_int(npang))

for i in range(npang):
    grid[i,4] = 4.0*numpy.pi*grids[i,3]
    rxy = grids[i,0]*grids[i,0] + grids[i,1]*grids[i,1]
    r = numpy.sqrt(rxy + grids[i,2]*grids[i,2])
    if (rxy < EPS):
        if (grids[i,2] >= 0.0):
            grid[i,0] = +1.0
        else:
            grid[i,0] = -1.0
        grid[i,1] = 0.0
        grid[i,3] = 0.0
        grid[i,2] = 1.0
    else:
        rxy = numpy.sqrt(rxy)
        grid[i,0] = grids[i,2]/r
        grid[i,1] = numpy.sqrt((1.0-grid[i,0])*(1.0+grid[i,0]))
        grid[i,2] = grids[i,0]/rxy
        grid[i,3] = grids[i,1]/rxy

del(grids)        

a = [0.0, 0.1, 1.0]
a = numpy.asarray(a)
a = numpy.reshape(a, (-1,3))
t0 = time.clock()
ao = dft.numint.eval_ao(mol, a, deriv=1)
rho = dft.numint.eval_rho2(mol, ao, mo_coeff, mo_occ, xctype='GGA')
log.timer('own', t0)
print "rho info : ", rho

feval = 'surf_driver'
drv = getattr(libaim, feval)
ct = numpy.asarray(grid[:,0], order='C')
st = numpy.asarray(grid[:,1], order='C')
cp = numpy.asarray(grid[:,2], order='C')
sp = numpy.asarray(grid[:,3], order='C')

drv(ctypes.c_int(inuc), 
    ctypes.c_int(npang), 
    ct.ctypes.data_as(ctypes.c_void_p),
    st.ctypes.data_as(ctypes.c_void_p),
    cp.ctypes.data_as(ctypes.c_void_p),
    sp.ctypes.data_as(ctypes.c_void_p),
    ctypes.c_int(ntrial), rpru.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(epsiscp),
    ctypes.c_double(epsroot), ctypes.c_double(rmaxsurf), ctypes.c_int(backend),
    ctypes.c_double(epsilon), ctypes.c_double(step), ctypes.c_int(mstep),
    ctypes.c_int(cart),
    coords.ctypes.data_as(ctypes.c_void_p),
    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
    bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
    env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nprim),
    ao_loc.ctypes.data_as(ctypes.c_void_p),
    mo_coeff.ctypes.data_as(ctypes.c_void_p),
    mo_occ.ctypes.data_as(ctypes.c_void_p))

del(ct,st,sp,cp)

