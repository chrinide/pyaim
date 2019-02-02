#!/usr/bin/env python

import time, numpy, x2cmp2
from pyscf import lib, gto, scf, x2c, ao2mo
from pyscf.df import r_incore

#At 0.0 0.0  0.000
#At 0.0 0.0  3.100
#Pb 0.0 0.0 0.00
#O  0.0 0.0 1.922
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

t = time.time()
cderi = r_incore.cholesky_eri(mol, int3c='int3c2e_spinor', verbose=4)
def fjk2c(mol, dm, *args, **kwargs):
    n2c = dm.shape[0]
    cderi_ll = cderi.reshape(-1,n2c,n2c)
    vj = numpy.zeros((n2c,n2c), dtype=dm.dtype)
    vk = numpy.zeros((n2c,n2c), dtype=dm.dtype)
    rho = (numpy.dot(cderi, dm.T.reshape(-1)))
    vj = numpy.dot(rho, cderi).reshape(n2c,n2c)
    v1 = lib.einsum('pij,jk->pik', cderi_ll, dm)
    vk = lib.einsum('pik,pkj->ij', v1, cderi_ll)
    return vj, vk

mf = x2c.RHF(mol)
dm = mf.get_init_guess() + 0.0j
mf.get_jk = fjk2c
mf.direct_scf = False
ehf = mf.scf(dm)
print('Time %.3f (sec)' % (time.time()-t))
     
ncore = 2
nao, nmo = mf.mo_coeff.shape
nocc = mol.nelectron - ncore
nvir = nmo - nocc - ncore
mo_core = mf.mo_coeff[:,:ncore]
mo_occ = mf.mo_coeff[:,ncore:ncore+nocc]
mo_vir = mf.mo_coeff[:,ncore+nocc:]
co = mo_occ
cv = mo_vir
ec = mf.mo_energy[:ncore]
eo = mf.mo_energy[ncore:ncore+nocc]
ev = mf.mo_energy[ncore+nocc:]
lib.logger.info(mf,"* Core orbitals: %d" % ncore)
lib.logger.info(mf,"* Virtual orbitals: %d" % (len(ev)))

n2c = mol.nao_2c()
dferi = cderi.reshape(-1,n2c,n2c)

#eri_ao = lib.einsum('Qia,Qjb->iajb', dferi, dferi) 
#eri_mo = ao2mo.general(eri_ao,(co,cv,co,cv)).reshape(nocc,nvir,nocc,nvir) 
#eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
#e_denom = 1.0/(eo.reshape(-1,1,1,1)-ev.reshape(-1,1,1)+eo.reshape(-1,1)-ev)
#t2 = eri_mo.conj()*e_denom
#e_mp2 = numpy.einsum('iajb,iajb', t2, eri_mo)*0.25
#lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
#lib.logger.info(mf,"!**** E(HF+MP2): %s" % (e_mp2+ehf))

#eri_mo = lib.einsum('rj,Qrs->Qjs', co.conj(), dferi)
#eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
#eri_mo = lib.einsum('Qia,Qjb->iajb', eri_mo, eri_mo) 
#eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
#t2 = eri_mo.conj()*e_denom
#e_mp2 = numpy.einsum('iajb,iajb', t2, eri_mo)*0.25
#lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
#lib.logger.info(mf,"!**** E(HF+MP2): %s" % (e_mp2+ehf))

t2 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=numpy.complex128)
t = time.time()
eri_mo = lib.einsum('rj,Qrs->Qjs', co.conj(), dferi)
eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
e_mp2 = 0.0
vv_denom = -ev.reshape(-1,1)-ev
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
        v = viajb - vibja
        div = 1.0/(eps_i + eps_j + vv_denom)
        e_mp2 += 0.25*numpy.einsum('ab,ab->', v, v.conj()*div) 
        t2[i,:,j,:] += v.conj()*div

lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %s" % (e_mp2+ehf))
print('Time %.3f (sec)' % (time.time()-t))

#eri_mo = lib.einsum('Qia,Qjb->iajb', eri_mo, eri_mo) 
#eri_mo = eri_mo - eri_mo.transpose(0,3,2,1)
#e_mp2 = numpy.einsum('iajb,iajb', t2, eri_mo)*0.25
#lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
#lib.logger.info(mf,"!**** E(HF+MP2): %s" % (e_mp2+ehf))

t = time.time()
dab = numpy.zeros((len(ev),len(ev)), dtype=numpy.complex128)
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
        v = viajb - vibja
        div = 1.0/(eps_i + eps_j + vv_denom)
        t2ij = v.conj()*div
        dab += lib.einsum('ea,eb->ab', t2ij,t2ij.conj())*0.5
    
dab = dab + dab.conj().T
dab *= 0.5
natoccvir, natorbvir = numpy.linalg.eigh(-dab)
for i, k in enumerate(numpy.argmax(abs(natorbvir), axis=0)):
    if natorbvir[k,i] < 0:
        natorbvir[:,i] *= -1
natoccvir = -natoccvir
lib.logger.debug(mf,"* Occupancies")
lib.logger.debug(mf,"* %s" % natoccvir)
lib.logger.debug(mf,"* The sum is %8.6f" % numpy.sum(natoccvir)) 
thresh_vir = 1e-4
active = (thresh_vir <= natoccvir)
lib.logger.debug(mf,"* Natural Orbital selection")
for i in range(nvir):
    lib.logger.debug(mf,"orb: %d %s %8.6f" % (i,active[i],natoccvir[i]))
    actIndices = numpy.where(active)[0]
lib.logger.info(mf,"* Original active orbitals %d" % nvir)
lib.logger.info(mf,"* Virtual core orbitals: %d" % (nvir-len(actIndices)))
lib.logger.info(mf,"* New active orbitals %d" % len(actIndices))
lib.logger.debug(mf,"* Active orbital indices %s" % actIndices)
natorbvir = natorbvir[:,actIndices]                                    
fvv = numpy.diag(ev)
fvv = reduce(numpy.dot, (natorbvir.conj().T, fvv, natorbvir))
ev, fnov = numpy.linalg.eigh(fvv)
cv = reduce(numpy.dot, (cv, natorbvir, fnov))
coeff = numpy.hstack([mo_core,co,cv])
energy = numpy.hstack([ec,eo,ev]) 
occ = numpy.zeros(coeff.shape[1])
for i in range(mol.nelectron):
    occ[i] = 1.0
print('Time FNO %.3f (sec)' % (time.time()-t))

t = time.time()
eri_mo = lib.einsum('rj,Qrs->Qjs', co.conj(), dferi)
eri_mo = lib.einsum('sb,Qjs->Qjb', cv, eri_mo)
e_mp2 = 0.0
vv_denom = -ev.reshape(-1,1)-ev
for i in range(nocc):
    eps_i = eo[i]
    i_Qv = eri_mo[:, i, :].copy()
    for j in range(nocc):
        eps_j = eo[j]
        j_Qv = eri_mo[:, j, :].copy()
        viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
        vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
        v = viajb - vibja
        div = 1.0/(eps_i + eps_j + vv_denom)
        e_mp2 += 0.25*numpy.einsum('ab,ab->', v, v.conj()*div) 

lib.logger.info(mf,"!*** E(MP2): %s" % e_mp2)
lib.logger.info(mf,"!**** E(HF+MP2): %s" % (e_mp2+ehf))
print('Time %.3f (sec)' % (time.time()-t))

#pt = x2cmp2.GMP2(mf, mo_coeff=coeff, mo_occ=occ)
#pt.frozen = ncore
#pt.kernel(mo_energy=energy)

#t = time.time()
#pt = x2cmp2.GMP2(mf)
#pt.frozen = ncore
#pt.kernel()
#print('Time %.3f (sec)' % (time.time()-t))

import x2cccsd
mycc = x2cccsd.GCCSD(mf, mo_coeff=coeff, mo_occ=occ)
mycc.frozen = ncore
ecc, t1, t2 = mycc.kernel()
et = mycc.ccsd_t()
print('(T) correlation energy', et)
print('CCSD(T) correlation energy', mycc.e_corr + et)

