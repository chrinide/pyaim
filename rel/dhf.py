#!/usr/bin/env python

def eval_ao(mol, coords, deriv=0):

    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)

    ao = mol.eval_gto('GTOval_sp_spinor', coords, 1, shls_slice, non0tab)
    if (deriv == 0):
        ngrid, nao = aoLa.shape[-2:]
        aoSa = numpy.ndarray((1,ngrid,nao), dtype=numpy.complex128)
        aoSb = numpy.ndarray((1,ngrid,nao), dtype=numpy.complex128)
        aoSa[0] = ao[0]
        aoSb[0] = ao[1]
    elif (deriv == 1):
        ngrid, nao = aoLa[0].shape[-2:]
        aoSa = numpy.ndarray((4,ngrid,nao), dtype=numpy.complex128)
        aoSb = numpy.ndarray((4,ngrid,nao), dtype=numpy.complex128)
        aoSa[0] = ao[0]
        aoSb[0] = ao[1]
        ao = mol.eval_gto('GTOval_ipsp_spinor', coords, 3, shls_slice, non0tab)
        for k in range(1,4):
            aoSa[k,:,:] = ao[0,k-1,:,:]
            aoSb[k,:,:] = ao[1,k-1,:,:]

    if deriv == 0:
        aoSa = aoSa[0]
        aoSb = aoSb[0]

    return aoLa, aoLb, aoSa, aoSb

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
        for i in range(1,4):
            rho[i] += numpy.einsum('pi,pi->p', aoa[i].real, c0a.real)
            rho[i] += numpy.einsum('pi,pi->p', aoa[i].imag, c0a.imag)
            rho[i] += numpy.einsum('pi,pi->p', aob[i].real, c0b.real)
            rho[i] += numpy.einsum('pi,pi->p', aob[i].imag, c0b.imag)
            rho[i] *= 2 

    return rho
    
OCCDROP = 1e-6
def eval_rho2(mol, ao, mo_coeff, mo_occ, small=False, xctype='LDA'):

    aoa, aob = ao
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = aoa. shape[-2:]
    else:
        ngrids, nao = aoa[0].shape[-2:]

    if (small == True):
        cpos = mo_coeff[n2c:,n2c:n2c+mol.nelectron] * c1
    else:
        pos = mo_occ > OCCDROP
        cpos = mo_coeff[:n2c,pos]

    if (xctype == 'LDA'):
        c0a = lib.dot(aoa, cposa)
        rhoaa = numpy.einsum('pi,pi->p', c0a.real, c0a.real)
        rhoaa += numpy.einsum('pi,pi->p', c0a.imag, c0a.imag)
        c0b = lib.dot(aob, cposb)
        rhobb = numpy.einsum('pi,pi->p', c0b.real, c0b.real)
        rhobb += numpy.einsum('pi,pi->p', c0b.imag, c0b.imag)
        rho = (rhoaa + rhobb)
    elif xctype == 'GGA':
        rho = numpy.zeros((4,ngrids))
        c0a = lib.dot(aoa[0], cpos)
        rhoaa = numpy.einsum('pi,pi->p', c0a.real, c0a.real)
        rhoaa += numpy.einsum('pi,pi->p', c0a.imag, c0a.imag)
        c0b = lib.dot(aob[0], cpos)
        rhobb = numpy.einsum('pi,pi->p', c0b.real, c0b.real)
        rhobb += numpy.einsum('pi,pi->p', c0b.imag, c0b.imag)
        rho[0] = (rhoaa + rhobb)
        for i in range(1,4):
            c1a = lib.dot(aoa[i], cpos)
            rho[i] += numpy.einsum('pi,pi->p', c0a.real, c1a.real)*2 # *2 for +c.c.
            rho[i] += numpy.einsum('pi,pi->p', c0a.imag, c1a.imag)*2 # *2 for +c.c.
            c1b = lib.dot(aob[i], cpos)
            rho[i] += numpy.einsum('pi,pi->p', c0b.real, c1b.real)*2 # *2 for +c.c.
            rho[i] += numpy.einsum('pi,pi->p', c0b.imag, c1b.imag)*2 # *2 for +c.c.

    return rho

import numpy
from pyscf import gto, scf, lib, dft

name = 'dhf'

mol = gto.Mole()
mol.basis = 'unc-dzp-dk'
mol.atom = '''
O      0.000000      0.000000      0.118351
H      0.000000      0.761187     -0.469725
H      0.000000     -0.761187     -0.469725
'''
mol.charge = 0
mol.spin = 0
mol.symmetry = 1
mol.verbose = 4
mol.nucmod = 1
mol.build()

mf = scf.DHF(mol)
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.chkfile = name+'.chk'
mf.kernel()

grids = dft.gen_grid.Grids(mol)
grids.kernel()
dm = mf.make_rdm1()
coords = grids.coords
weights = grids.weights

nao = mf.mo_occ.shape
n2c = mol.nao_2c()
c1 = 0.5/lib.param.LIGHT_SPEED
dmLL = dm[:n2c,:n2c].copy('C')
dmSS = dm[n2c:,n2c:] * c1**2

#rho += rhoS
#M = |\beta\Sigma|
#m[0] -= mS[0]
#m[1] -= mS[1]
#m[2] -= mS[2]
#s = lib.norm(m, axis=0)
#rhou = (r + s) * .5
#rhod = (r - s) * .5
#rho = (rhou, rhod)

aoLS = eval_ao(mol, coords, deriv=1)
rho = eval_rho(mol, aoLS[:2], dmLL, xctype='GGA')
rhoS = eval_rho(mol, aoLS[2:], dmSS, xctype='GGA')
print('RhoL = %.12f' % numpy.dot(rho[0], weights))
print('RhoS = %.12f' % numpy.dot(rhoS[0], weights))
print('Rho = %.12f' % numpy.dot(rho[0]+rhoS[0], weights))

coords = numpy.zeros(3)
coords = coords.reshape(-1,3)
aoLS = eval_ao(mol, coords, deriv=1)
rho = eval_rho(mol, aoLS[:2], dmLL, xctype='GGA')
rhoS = eval_rho(mol, aoLS[2:], dmSS, xctype='GGA')
print rho
print rhoS
print rho+rhoS
rho2 = eval_rho2(mol, aoLS[:2], mf.mo_coeff, mf.mo_occ, small=False, xctype='GGA')
rho2S = eval_rho2(mol, aoLS[2:], mf.mo_coeff, mf.mo_occ, small=True, xctype='GGA')
print rho2
print rho2S
print rho2+rho2S

