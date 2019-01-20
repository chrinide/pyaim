#!/usr/bin/env python

import h5py
import numpy
import sys

from pyscf import lib
from pyscf.tools.dump_mat import dump_tri

log = lib.logger.Logger(sys.stdout, 4)
    
name = 'x2c.chk'
atm = [0,1]
nmo = 10
aom1 = numpy.zeros((nmo,nmo))
aom2 = numpy.zeros((nmo,nmo))

with h5py.File(name+'.h5') as f:
    idx = 'ovlp'+str(atm[0])
    aom1 = f[idx+'/aom'].value
    idx = 'ovlp'+str(atm[1])
    aom2 = f[idx+'/aom'].value

delta = 2*numpy.einsum('ij,ji->', aom1, aom2)
log.info('Delta %f for pair %d %d' %  (delta, atm[0], atm[1]))

#TODO:descompose delta in the basis of dafh which is diagonal

mol = lib.chkfile.load_mol(name)
mo_coeff = lib.chkfile.load(name, 'scf/mo_coeff')
mo_coeff = mo_coeff[:,0:nmo]

from pyscf.tools import molden
natocc, natorb = numpy.linalg.eigh(aom1)
log.info('Occ for DAFH %s', natocc)
log.info('Sum Occ for DAFH %f', natocc.sum())
natorb = numpy.dot(mo_coeff, natorb)

#def lowdin(s):
#    e, v = numpy.linalg.eigh(s)
#    return numpy.dot(v/numpy.sqrt(e), v.T.conj())
#s = mol.intor('int1e_ovlp_spinor')
#s12inv = lowdin(s)
#natorb = numpy.dot(s12inv,natorb)

with open('x2c.molden', 'w') as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, natorb, occ=natocc)

def eval_ao(mol, coords, deriv=0):

    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)
    print aoLa.shape, aoLb.shape
    return aoLa, aoLb

def eval_rho(mol, ao, dm, xctype='LDA'):

    aoa, aob = ao
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = aoa.shape[-2:]
    else:
        ngrids, nao = aoa[0].shape[-2:]

    if xctype == 'LDA':
        out = lib.dot(aoa, dm)
        rhoaa = numpy.einsum('pi,pi->p', aoa.real, out.real)
        rhoaa += numpy.einsum('pi,pi->p', aoa.imag, out.imag)
        out = lib.dot(aob, dm)
        rhobb = numpy.einsum('pi,pi->p', aob.real, out.real)
        rhobb += numpy.einsum('pi,pi->p', aob.imag, out.imag)
        rho = (rhoaa + rhobb)
    elif xctype == 'GGA':
        rho = numpy.zeros((4,ngrids))
        c0a = lib.dot(aoa[0], dm)
        rhoaa = numpy.einsum('pi,pi->p', aoa[0].real, c0a.real)
        rhoaa += numpy.einsum('pi,pi->p', aoa[0].imag, c0a.imag)
        c0b = lib.dot(aob[0], dm)
        rhobb = numpy.einsum('pi,pi->p', aob[0].real, c0b.real)
        rhobb += numpy.einsum('pi,pi->p', aob[0].imag, c0b.imag)
        rho[0] = (rhoaa + rhobb)
        for i in range(1, 4):
            rho[i] += numpy.einsum('pi,pi->p', aoa[i].real, c0a.real)
            rho[i] += numpy.einsum('pi,pi->p', aoa[i].imag, c0a.imag)
            rho[i] += numpy.einsum('pi,pi->p', aob[i].real, c0b.real)
            rho[i] += numpy.einsum('pi,pi->p', aob[i].imag, c0b.imag)
            rho[i] *= 2 

    return rho

from pyscf import lib, dft

def make_rdm1(mo_coeff, mo_occ):
    mocc = mo_coeff[:,mo_occ>0]
    return lib.dot(mocc*mo_occ[mo_occ>0], mocc.T.conj())
dm = make_rdm1(natorb, natocc)
grids = dft.gen_grid.Grids(mol)
grids.kernel()
coords = grids.coords
weights = grids.weights
ao = eval_ao(mol, coords, deriv=1)
rho = eval_rho(mol, ao, dm, xctype='GGA')
print('Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))

