#!/usr/bin/env python

import h5py, numpy, sys, time
from pyscf import lib, dft
from pyscf.lib import logger
from pyaim import grid

log = lib.logger.Logger(sys.stdout, 4)
log.verbose = 5
lib.logger.TIMER_LEVEL = 5

name = 'test/lif'
betafac = 0.4
nrad = 101
iqudr = 1
surface = 'test/lif.chk.h5'
mol = lib.chkfile.load_mol(name+'.chk')
charges = mol.atom_charges()
coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in mol._atom])
natm = mol.natm		

log.info('Verbose level : %d' % log.verbose)
log.info('Checkpoint file is : %s' % (name+'.chk'))
log.info('Surface file is : %s' % surface)
log.info('Num atoms : %d ' % natm)
log.info('Num electrons : %d ' % mol.nelectron)
log.info('Total charge : %d ' % mol.charge)
log.info('Spin : %d ' % mol.spin)
log.info('Atom Coordinates')
for i in range(natm):
    log.info('Nuclei %d position : %8.5f %8.5f %8.5f', i, *coords[i])

mf = lib.chkfile.load(name+'.chk', 'scf')
mo_coeff = lib.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ = lib.chkfile.load(name+'.chk', 'scf/mo_occ')

xnuc = numpy.zeros(3)
xyzrho = numpy.zeros(3)
with h5py.File(surface) as f:
    rmin = f['atom0/rmin'].value
    rmax = f['atom0/rmax'].value
    xnuc = f['atom0/xnuc'].value
    xyzrho = f['atom0/xyzrho'].value
    npang = f['atom0/npang'].value
    ntrial = f['atom0/ntrial'].value
    rsurf = numpy.zeros((npang,ntrial))
    nlimsurf = numpy.zeros((npang))
    coordsang = numpy.zeros((npang,4))
    rsurf = f['atom0/surface'].value
    nlimsurf = f['atom0/intersecs'].value
    coordsang = f['atom0/coords'].value
brad = rmin*betafac

log.info('Nuclei position : %8.5f %8.5f %8.5f', *xnuc)
log.info('Nuclei rho position : %8.5f %8.5f %8.5f', *xyzrho)
log.info('Angular point : %d ', npang)
log.info('Radial point : %d ', nrad)
log.info('Ntrial : %d ', ntrial)
log.info('Rmin distance to ZFS : %8.5f ', rmin)
log.info('Rmax distance to ZFS : %8.5f ', rmax)
log.info('Beta rad : %8.5f ', brad)
log.info('Quadrature : legendre ')

# TODO:screaning of points
def rho(x):
    x = numpy.reshape(x, (-1,3))
    ao = dft.numint.eval_ao(mol, x, deriv=1)
    rho = dft.numint.eval_rho2(mol, ao, mo_coeff, mo_occ, xctype='GGA')
    return rho[0]

xcoor = numpy.zeros(3)

log.info('Go with inside betasphere')
r0 = 0
rfar = brad
rad = 0.41 #rbrag[]
mapr = 2
log.info('Quadrature mapping : %d ', mapr)
rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
rlmr = 0.0
t0 = time.clock()
coords = numpy.empty((npang,3))
for n in range(nrad):
    r = rmesh[n]
    rlm = 0.0
    for j in range(npang): # j-loop can be changed to map
        cost = coordsang[j,0]
        sintcosp = coordsang[j,1]*coordsang[j,2]
        sintsinp = coordsang[j,1]*coordsang[j,3]
        xcoor[0] = r*sintcosp
        xcoor[1] = r*sintsinp
        xcoor[2] = r*cost    
        p = xnuc + xcoor
        coords[j] = p
    den = rho(coords)
    rlm = numpy.einsum('i,i->', den, coordsang[:,4])
    rlmr += rlm*dvol[n]*rwei[n]
log.info('Electron density inside bsphere %8.5f ', rlmr)    
log.timer('Bsphere build', t0)
rhob = rlmr
################
log.info('Go outside betasphere')
EPS = 1e-6
def inbasin(r,j):

    isin = False
    rs1 = 0.0
    irange = nlimsurf[j]
    irange = int(irange)
    for k in range(irange):
        rs2 = rsurf[j,k]
        if (r >= rs1-EPS and r <= rs2+EPS):
            if (((k+1)%2) == 0):
                isin = False
            else:
                isin = True
            return isin
        rs1 = rs2

    return isin

r0 = brad
rfar = rmax
rad = 0.41 #rbrag[]
mapr = 2
log.info('Quadrature mapping : %d ', mapr)
rmesh, rwei, dvol, dvoln = grid.rquad(nrad,r0,rfar,rad,iqudr,mapr)
rlmr = 0.0
t0 = time.clock()
for n in range(nrad):
    r = rmesh[n]
    rlm = 0.0
    coords = []
    weigths = []
    for j in range(npang):
        inside = True
        inside = inbasin(r,j)
        if (inside == True):
            cost = coordsang[j,0]
            sintcosp = coordsang[j,1]*coordsang[j,2]
            sintsinp = coordsang[j,1]*coordsang[j,3]
            xcoor[0] = r*sintcosp
            xcoor[1] = r*sintsinp
            xcoor[2] = r*cost    
            p = xnuc + xcoor
            coords.append(p)
            weigths.append(coordsang[j,4])
    coords = numpy.array(coords)
    weigths = numpy.array(weigths)
    den = rho(coords)
    rlm = numpy.einsum('i,i->', den, weigths)
    rlmr += rlm*dvol[n]*rwei[n]
log.info('Electron density outside bsphere %8.5f ', rlmr)    
log.timer('Out Bsphere build', t0)
rhoo = rlmr
log.info('Electron density %8.5f ', (rhob+rhoo))    

