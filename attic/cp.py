#!/usr/bin/env python

import numpy, sys, time
from pyscf import lib, dft
from pyscf.lib import logger

log = lib.logger.Logger(sys.stdout, 4)
log.verbose = 5
lib.logger.TIMER_LEVEL = 5

name = 'h2o'
mol = lib.chkfile.load_mol(name+'.chk')
charges = mol.atom_charges()
coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in mol._atom])
natm = mol.natm		

log.info('Verbose level : %d' % log.verbose)
log.info('Checkpoint file is : %s' % (name+'.chk'))
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

import hess
HMINIMAL = numpy.finfo(numpy.float64).eps
def rhohess(x):
    x = numpy.reshape(x, (-1,3))
    hessi = numpy.zeros((3,3))
    rho = hess.eval_rho3(mol, x, mo_coeff, mo_occ, deriv=2)
    gradmod = numpy.linalg.norm(rho[1:4:,0])
    hessi[0,0] = rho[6,0]
    hessi[0,1] = rho[7,0] 
    hessi[0,2] = rho[8,0] 
    hessi[1,0] = rho[7,0] 
    hessi[1,1] = rho[9,0] 
    hessi[1,2] = rho[10,0] 
    hessi[2,0] = rho[8,0] 
    hessi[2,1] = rho[10,0] 
    hessi[2,2] = rho[11,0] 
    return rho[0,0], rho[1:4:,0]/(gradmod+HMINIMAL), gradmod, hessi, rho[4]

# Find critical points of electron density
NEWTONEPS = 1e-5
#def newton(point):
#     call fpoint(r)
#     it = 0
#     ier=0
#     r1 = numpy.zeros(3)
#     while (it < maxit and gradmod >= eps)
#       it = it + 1
#       hess = numpy.linalg.inv(hess)
#       do j = 1,3
#         do k = 1,3
#           r1(j) += hess(j,k)*grad(k)
#       point(j) = point(j) - r1(j)
#       call fpoint(r)
#     if (it.ge.maxit) ier=1
