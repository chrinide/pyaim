#!/usr/bin/env python

def eval_ao(mol, coords, deriv=0):

    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)
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

import numpy
from pyscf import gto, scf, x2c, lib, dft

name = 'x2c'

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 0
mol.verbose = 4
mol.build()

mf = x2c.UHF(mol)
mf.chkfile = name+'.chk'
mf.kernel()

#x = mf.with_x2c.get_xmat()

#dm = mf.make_rdm1(mo_coeff, mo_occ)
dm = mf.make_rdm1()
a = numpy.zeros(3)
a = a.reshape(-1,3)
ao = eval_ao(mol, a, deriv=1)
rho = eval_rho(mol, ao, dm, xctype='GGA')
print rho

