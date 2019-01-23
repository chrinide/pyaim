#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib, dft
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'puo2_+2.chk'
atm = [0,1]

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1, aom2.conj())
log.info('Delta %f for pair %d %d' %  (delta.real, atm[0], atm[1]))

occdrop = 1e-12
mol = lib.chkfile.load_mol(name)
mo_coeff = lib.chkfile.load(name, 'scf/mo_coeff')
mo_occ = lib.chkfile.load(name, 'scf/mo_occ')
nocc = mo_occ[abs(mo_occ)>occdrop]
nocc = len(nocc)
pos = abs(mo_occ) > occdrop
n2c = mol.nao_2c()
mo_coeffL = mo_coeff[:n2c,pos]
cspeed = lib.param.LIGHT_SPEED
c1 = 0.5/cspeed
mo_coeffS = mo_coeff[n2c:,n2c:n2c+nocc]*c1

dab = 2*numpy.einsum('ik,kj->ij', aom1, aom2.conj())
dba = 2*numpy.einsum('ik,kj->ij', aom2, aom1.conj())
d2c = (dab+dba)/2.0
natocc, natorb = numpy.linalg.eigh(d2c)
log.info('Occ for NADO %s', natocc)
log.info('Sum Occ for NADO %f', natocc.sum())

natorbL = numpy.dot(mo_coeffL, natorb)
natorbS = numpy.dot(mo_coeffS, natorb)

def eval_ao(mol, coords, deriv=0):
    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)
    feval = 'GTOval_sp_spinor'
    aoSa, aoSb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)
    return aoLa, aoLb, aoSa, aoSb

def eval_rho(mol, ao, dmL, dmS):
    aoLa, aoLb, aoSa, aoSb = ao
    ngrids, nao = aoLa.shape[-2:]

    out = lib.dot(aoLa, dmL)
    rhoaa = numpy.einsum('pi,pi->p', aoLa.real, out.real)
    rhoaa += numpy.einsum('pi,pi->p', aoLa.imag, out.imag)
    out = lib.dot(aoLb, dmL)
    rhobb = numpy.einsum('pi,pi->p', aoLb.real, out.real)
    rhobb += numpy.einsum('pi,pi->p', aoLb.imag, out.imag)
    rho = (rhoaa + rhobb)

    out = lib.dot(aoSa, dmS)
    rhoaa = numpy.einsum('pi,pi->p', aoSa.real, out.real)
    rhoaa += numpy.einsum('pi,pi->p', aoSa.imag, out.imag)
    out = lib.dot(aoSb, dmS)
    rhobb = numpy.einsum('pi,pi->p', aoSb.real, out.real)
    rhobb += numpy.einsum('pi,pi->p', aoSb.imag, out.imag)
    rho += (rhoaa + rhobb)

    return rho

def make_rdm1(mo_coeff, mo_occ):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

dmL = make_rdm1(natorbL, natocc)
dmS = make_rdm1(natorbS, natocc)
grids = dft.gen_grid.Grids(mol)
grids.verbose = 0
grids.kernel()
coords = grids.coords
weights = grids.weights
ao = eval_ao(mol, coords, deriv=0)
rho = eval_rho(mol, ao, dmL, dmS)
print('Rho = %.12f' % numpy.dot(rho, weights))

