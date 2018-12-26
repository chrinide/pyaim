#!/usr/bin/env python

import numpy, sys, time, ctypes, os
from pyscf import lib, dft
from pyscf.lib import logger

_loaderpath = os.path.dirname(__file__)
libaim = numpy.ctypeslib.load_library('libaim.so', _loaderpath)

EPS = 1e-7
libcgto = lib.load_library('libcgto')
libdft = lib.load_library('libdft')

name = 'lif'
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
nprim, nmo = mo_coeff.shape
ao_loc = mol.ao_loc_nr()

log = lib.logger.Logger(sys.stdout, 4)
lib.logger.TIMER_LEVEL = 5
log.verbose = 5

log.info('Verbose level : %d' % log.verbose)
log.info('Checkpoint file is : %s' % (name+'.chk'))
log.info('Num atoms : %d ' % natm)
log.info('Num electrons : %d ' % mol.nelectron)
log.info('Total charge : %d ' % mol.charge)
log.info('Spin : %d ' % mol.spin)
log.info('Atom Coordinates')
for i in range(natm):
    log.info('Nuclei %d position : %8.5f %8.5f %8.5f', i, *coords[i])

inuc = 0
xnuc = coords[inuc]
xyzrho = xnuc
epsiscp = 0.180
ntrial = 11
npang = 6
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

log.info('Looking surface for atom : %d', inuc)
log.info('Nuclei position : %8.5f %8.5f %8.5f', *xnuc)
log.info('Nuclei rho position : %8.5f %8.5f %8.5f', *xyzrho)
log.info('Lebedev angular points : %d ', npang)
log.info('Ntrial : %d ', ntrial)

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

feval = 'surf_driver'
drv = getattr(libaim, feval)
ct = numpy.asarray(grid[:,0], order='C')
st = numpy.asarray(grid[:,1], order='C')
cp = numpy.asarray(grid[:,2], order='C')
sp = numpy.asarray(grid[:,3], order='C')

rsurf = numpy.zeros((npang,ntrial), order='C')
nlimsurf = numpy.zeros((npang), order='C', dtype=numpy.int32)

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
    coords.ctypes.data_as(ctypes.c_void_p), xyzrho.ctypes.data_as(ctypes.c_void_p),
    atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
    bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
    env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nprim), ctypes.c_int(nmo), 
    ao_loc.ctypes.data_as(ctypes.c_void_p),
    mo_coeff.ctypes.data_as(ctypes.c_void_p),
    mo_occ.ctypes.data_as(ctypes.c_void_p),
    nlimsurf.ctypes.data_as(ctypes.c_void_p),
    rsurf.ctypes.data_as(ctypes.c_void_p))

for i in range(npang):
    print(i,ct[i],st[i],sp[i],cp[i],nlimsurf[i],rsurf[i,:nlimsurf[i]])

rmin = 1000.0
rmax = 0.0
for i in range(npang):
    nsurf = nlimsurf[i]
    rmin = numpy.minimum(rmin,rsurf[i,0])
    rmax = numpy.maximum(rmax,rsurf[i,nsurf-1])
log.info('Rmin for surface : %8.5f ', rmin)
log.info('Rmax for surface : %8.5f ', rmax)

atom_dic = {'inuc':inuc,
            'xnuc':xnuc,
            'xyzrho':xyzrho,
            'coords':grid,
            'intersecs':nlimsurf,
            'surface':rsurf,
            'npang':npang,
            'rmin':rmin,
            'rmax':rmax,
            'ntrial':ntrial}
lib.chkfile.save('test.h5', 'atom'+str(inuc), atom_dic)

del(ct,st,sp,cp,rsurf,nlimsurf,grid)

