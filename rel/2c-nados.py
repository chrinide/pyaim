#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'puo2_+2_x2c.chk'
atm = [0,1]
nmo = 108

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1, aom2.conj())
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

#TODO:descompose delta in the basis of dafh which is diagonal

mol = lib.chkfile.load_mol(name)
mo_coeff = lib.chkfile.load(name, 'scf/mo_coeff')
mo_coeff = mo_coeff[:,0:nmo]

dab = 2*numpy.einsum('ik,kj->ij', aom1, aom2.conj())
dba = 2*numpy.einsum('ik,kj->ij', aom2, aom1.conj())
d2c = (dab+dba)/2.0

natocc, natorb = numpy.linalg.eigh(d2c)
log.info('Occ for NADO %s', natocc)
log.info('Sum Occ for NADO %f', natocc.sum())
natorb = numpy.dot(mo_coeff, natorb)

def eval_ao(mol, coords, deriv=0):
    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)
    return aoLa, aoLb

def eval_rho(mol, ao, dm):
    aoa, aob = ao
    ngrids, nao = aoa.shape[-2:]
    out = lib.dot(aoa, dm)
    rhoaa = numpy.einsum('pi,pi->p', aoa.real, out.real)
    rhoaa += numpy.einsum('pi,pi->p', aoa.imag, out.imag)
    out = lib.dot(aob, dm)
    rhobb = numpy.einsum('pi,pi->p', aob.real, out.real)
    rhobb += numpy.einsum('pi,pi->p', aob.imag, out.imag)
    rho = (rhoaa + rhobb)
    return rho

def make_rdm1(mo_coeff, mo_occ):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

dm = make_rdm1(natorb, natocc)
grids = dft.gen_grid.Grids(mol)
grids.verbose = 0
grids.kernel()
coords = grids.coords
weights = grids.weights
ao = eval_ao(mol, coords, deriv=0)
rho = eval_rho(mol, ao, dm)
print('Rho = %.12f' % numpy.dot(rho, weights))

