#!/usr/bin/env python

def eval_ao(mol, coords, deriv=0, with_s=True):

    non0tab = None
    shls_slice = None
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    feval = 'GTOval_spinor_deriv%d' % deriv
    aoLa, aoLb = mol.eval_gto(feval, coords, comp, shls_slice, non0tab)

    if with_s:
        ao = mol.eval_gto('GTOval_sp_spinor', coords, 1, shls_slice, non0tab)
        if (deriv == 0):
            ngrid, nao = aoLa.shape[-2:]
            aoSa = numpy.ndarray((comp,ngrid,nao), dtype=numpy.complex128)
            aoSb = numpy.ndarray((comp,ngrid,nao), dtype=numpy.complex128)
            aoSa[0] = ao[0]
            aoSb[0] = ao[1]
        elif (deriv == 1):
            ngrid, nao = aoLa[0].shape[-2:]
            aoSa = numpy.ndarray((comp,ngrid,nao), dtype=numpy.complex128)
            aoSb = numpy.ndarray((comp,ngrid,nao), dtype=numpy.complex128)
            aoSa[0] = ao[0]
            aoSb[0] = ao[1]
            comp = 3
            ao = mol.eval_gto('GTOval_ipsp_spinor', coords, comp, shls_slice, non0tab)
            for k in range(1,4):
                aoSa[k,:,:] = ao[0,k-1,:,:]
                aoSb[k,:,:] = ao[1,k-1,:,:]
        else :
            raise RuntimeError('eval_ao no available')

        if deriv == 0:
            aoSa = aoSa[0]
            aoSb = aoSb[0]

    return aoLa, aoLb, aoSa, aoSb

#TODO: \nabla^2 rho and tau = 1/2 (\nabla f)^2
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
        rhoaa+= numpy.einsum('pi,pi->p', aoa.imag, out.imag)
        #rhoba = numpy.einsum('pi,pi->p', aob.real, out.real)
        #rhoba+= numpy.einsum('pi,pi->p', aob.imag, out.imag)
        out = lib.dot(aob, dm)
        #rhoab = numpy.einsum('pi,pi->p', aoa.real, out.real)
        #rhoab+= numpy.einsum('pi,pi->p', aoa.imag, out.imag)
        rhobb = numpy.einsum('pi,pi->p', aob.real, out.real)
        rhobb+= numpy.einsum('pi,pi->p', aob.imag, out.imag)
        rho = (rhoaa + rhobb).real
        #mx = rhoab + rba
        #my =(rhoba - rhoab)*1j
        #mz = rhoaa - rhobb
        #m = numpy.vstack((mx, my, mz))
    elif xctype == 'GGA':
        rho = numpy.empty((4,ngrids))
        c0a = lib.dot(aoa[0], dm)
        rhoaa = numpy.einsum('pi,pi->p', aoa[0].real, c0a.real)
        rhoaa+= numpy.einsum('pi,pi->p', aoa[0].imag, c0a.imag)
        c0b = lib.dot(aob[0], dm)
        rhobb = numpy.einsum('pi,pi->p', aob[0].real, c0b.real)
        rhobb+= numpy.einsum('pi,pi->p', aob[0].imag, c0b.imag)
        rho[0] = (rhoaa + rhobb).real
        #for i in range(1, 4):
        #    rho[i] = numpy.einsum('pi,pi->p', aoa[i].real, c0a.real)
        #    rho[i]+= numpy.einsum('pi,pi->p', aoa[i].imag, c0a.imag)
        #    rho[i] = numpy.einsum('pi,pi->p', aob[i].real, c0b.real)
        #    rho[i]+= numpy.einsum('pi,pi->p', aob[i].imag, c0b.imag)
        #    rho[i] *= 2 # *2 for +c.c. in the next two lines
    else: # meta-GGA
        raise NotImplementedError

    return rho

#TODO: \nabla^2 rho and tau = 1/2 (\nabla f)^2
def eval_rho2(mol, ao, mo_coeff, mo_occ, small=False, xctype='LDA'):

    aoa, aob = ao
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = aoa.shape[-2:]
    else:
        ngrids, nao = aoa[0].shape[-2:]

    pos = mo_occ > OCCDROP
    if pos.sum() > 0:
        if (small == True):
            c1 = 0.5/lib.param.LIGHT_SPEED
            cposa = mo_coeff[nao:nao/2,pos]*c1**2
            cposb = mo_coeff[nao:,pos]*c1**2
        else:
            cposa = mo_coeff[0:nao/2,pos]
            cposb = mo_coeff[nao/2:nao,pos]
#   dmLL = dm[:n2c,:n2c].copy('C')
#   dmSS = dm[n2c:,n2c:] * c1**2
        if (xctype == 'LDA'):
            out = lib.dot(aoa, cposa)
            rhoaa = numpy.einsum('pi,pi->p', cposa.real, out.real)
            rhoaa+= numpy.einsum('pi,pi->p', cposa.imag, out.imag)
            out = lib.dot(aob, cposb)
            rhobb = numpy.einsum('pi,pi->p', cposb.real, out.real)
            rhobb+= numpy.einsum('pi,pi->p', cposb.imag, out.imag)
            rho = (rhoaa + rhobb).real
        else: 
            raise NotImplementedError

    else:
        if (xctype == 'LDA'):
            rho = numpy.zeros(ngrids)
        else:
            raise NotImplementedError


    return rho

import numpy
from pyscf import gto, scf, lib, dft

# Dimer in Bohr
#Au   0  0.0   0
#Au   0  4.67  0
mol = gto.M(
    #unit = 'B',
    atom = '''
H    0  0.0   0
H    0  0.77  0
''',
    basis = 'unc-dzp-dkh',
    verbose = 4,
    nucmod = 1,
    symmetry = 1,
)

mf = dft.DUKS(mol)
mf.with_ssss = True
mf.with_gaunt = False
mf.with_breit = False
mf.grids.radi_method = dft.mura_knowles
mf.grids.becke_scheme = dft.stratmann
mf.grids.level = 3
mf.grids.prune = None
mf.kernel()

#print mf.mo_occ.shape
#print mf.mo_occ

dm = mf.make_rdm1()
coords = mf.grids.coords
weights = mf.grids.weights
nao = mf.mo_occ.shape
n2c = mol.nao_2c()
#print 'n2c',n2c
with_s = (nao == n2c*2)  # 4C DM
if with_s:
    c1 = 0.5/lib.param.LIGHT_SPEED
    dmLL = dm[:n2c,:n2c].copy('C')
    dmSS = dm[n2c:,n2c:] * c1**2

aoLS = eval_ao(mol, coords, deriv=0)
if with_s:
#rho , m  = self.eval_rho(mol, ao[:2], dmLL[idm], non0tab, xctype)
#rhoS, mS = self.eval_rho(mol, ao[2:], dmSS[idm], non0tab, xctype)
#rho += rhoS
## M = |\beta\Sigma|
#m[0] -= mS[0]
#m[1] -= mS[1]
#m[2] -= mS[2]
#s = lib.norm(m, axis=0)
#rhou = (r + s) * .5
#rhod = (r - s) * .5
#rho = (rhou, rhod)
    rho = eval_rho(mol, aoLS[:2], dmLL, xctype='LDA')
    print('RhoL = %.12f' % numpy.einsum('i,i->', rho, weights))
    #print('RhoL = %.12f' % numpy.einsum('i,i->', rho[0], weights))
    rhoS = eval_rho(mol, aoLS[2:], dmSS, xctype='LDA')
    print('RhoS = %.12f' % numpy.einsum('i,i->', rhoS, weights))
    #print('RhoS = %.12f' % numpy.einsum('i,i->', rhoS[0], weights))
    rho += rhoS
    print('Rho = %.12f' % numpy.einsum('i,i->', rho, weights))
    #print('Rho = %.12f' % numpy.einsum('i,i->', rho[0], weights))
else:
    rho = eval_rho(mol, aoLS, dm)
    print('Rho = %.12f' % numpy.einsum('i,i->', rho, weights))

