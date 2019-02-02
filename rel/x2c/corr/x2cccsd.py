#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import addons
from pyscf.cc import gintermediates as imd
from pyscf import __config__
einsum = lib.einsum

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def update_amps(cc, t1, t2, eris):

    assert(isinstance(eris, _PhysicistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)
    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    eris_oovv = np.array(eris.oovv)
    e += 0.25*np.einsum('ijab,ijab', t2, eris_oovv)
    e += 0.5*np.einsum('ia,jb,ijab', t1, t1, eris_oovv)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in GCCSD energy %s', e)
    return e.real

class GCCSD(ccsd.CCSD):

    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        mo_e = eris.mo_energy
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = np.array(eris.oovv)
        t2 = eris_oovv / eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris_oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    energy = energy
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2=mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        e_corr, self.t1, self.t2 = ccsd.CCSD.ccsd(self, t1, t2, eris)
        return e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import gccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                gccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        import x2cccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return x2cccsd_t.kernel(self, eris, t1, t2, self.verbose)

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return gccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=False)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return gccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        mem_incore = nmo**4*2*4 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            logger.info(self,'Incore mem for 2e integrals')
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            logger.info(self,'Outcore mem for 2e integrals')
            return _make_eris_outcore(self, mo_coeff)

    def density_fit(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        raise NotImplementedError


class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = ccsd.get_frozen_mask(mycc)
        self.mo_coeff = mo_coeff[:,mo_idx]

        # Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        fockao = mycc._scf.get_fock(dm=dm)
        self.fock = reduce(np.dot, (self.mo_coeff.conj().T, fockao, self.mo_coeff))
        self.nocc = mycc.nocc

        mo_e = self.mo_energy = self.fock.diagonal().real
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small', gap)
        else:
            logger.info(mycc, 'HOMO-LUMO gap %s', gap)
        return self

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.clock(), time.time())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        mo = eris.mo_coeff
        eri = ao2mo.kernel(mycc.mol.intor('int2e_spinor'), mo).reshape(nmo,nmo,nmo,nmo)

    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)

    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    mo = eris.mo_coeff
    orbo = mo[:,:nocc]
    orbv = mo[:,nocc:]

    feri = eris.feri = lib.H5TmpFile()
    eris.oooo = feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'c8')
    eris.ooov = feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'c8')
    eris.oovv = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'c8')
    eris.ovov = feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'c8')
    eris.ovvo = feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'c8')
    eris.ovvv = feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'c8')
    eris.vvvv = feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'c8')

    max_memory = mycc.max_memory-lib.current_memory()[0]
    blksize = min(nocc, max(2, int(max_memory*1e6/8/(nmo**3*2))))
    max_memory = max(MEMORYMIN, max_memory)

    fswap = lib.H5TmpFile()
    ao2mo.kernel(mycc.mol, (mo,mo,mo,mo), fswap, 
                 max_memory=max_memory, verbose=log, intor='int2e_spinor') 

    for p0, p1 in lib.prange(0, nocc, blksize):
        tmp = np.asarray(fswap['eri_mo'][p0*nmo:p1*nmo])
        tmp = tmp.reshape(p1-p0,nmo,nmo,nmo)
        eris.oooo[p0:p1] = (tmp[:,:nocc,:nocc,:nocc].transpose(0,2,1,3) -
                     tmp[:nocc,:nocc,:nocc,:nocc].transpose(0,2,3,1))
        eris.ooov[p0:p1] = (tmp[:,:nocc,:nocc,nocc:].transpose(0,2,1,3) -
                     tmp[:nocc,nocc:,:nocc,:nocc].transpose(0,2,3,1))
        eris.ovvv[p0:p1] = (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3) -
                     tmp[:nocc,nocc:,nocc:,nocc:].transpose(0,2,3,1))
        eris.oovv[p0:p1] = (tmp[:,nocc:,:nocc,nocc:].transpose(0,2,1,3) -
                     tmp[:nocc,nocc:,:nocc,nocc:].transpose(0,2,3,1))
        eris.ovov[p0:p1] = (tmp[:,:nocc,nocc:,nocc:].transpose(0,2,1,3) -
                     tmp[:nocc,nocc:,nocc:,:nocc].transpose(0,2,3,1))
        eris.ovvo[p0:p1] = (tmp[:,nocc:,nocc:,:nocc].transpose(0,2,1,3) -
                     tmp[:nocc,:nocc,nocc:,nocc:].transpose(0,2,3,1))

    ao2mo.kernel(mycc.mol, (orbv,orbv,orbv,orbv), fswap, dataname='vvvv',
                 max_memory=max_memory, verbose=log, intor='int2e_spinor') 
    for p0, p1 in lib.prange(0, nvir, blksize):
        tmp = np.asarray(fswap['vvvv'][p0*nvir:p1*nvir])
        tmp = tmp.reshape(p1-p0,nvir,nvir,nvir)
        eris.vvvv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1)

    #tmp = np.asarray(fswap['eri_mo'])
    #tmp = tmp.reshape(nmo,nmo,nmo,nmo)
    #eris.vvvv = (tmp[nocc:,nocc:,nocc:,nocc:].transpose(0,2,1,3) -
    #             tmp[nocc:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
    cput0 = log.timer_debug1('transforming integrals', *cput0)

    return eris


if __name__ == '__main__':
    from pyscf import scf, x2c
    from pyscf import gto

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
    
    mf = x2c.RHF(mol)
    dm = mf.get_init_guess() + 0.1j
    mf.kernel(dm)

    ncore = 2
    mycc = GCCSD(mf)
    mycc.frozen = ncore
    ecc, t1, t2 = mycc.kernel()

    #import numpy
    #rdm1 = mycc.make_rdm1()
    #rdm2 = mycc.make_rdm2()
    #c = mf.mo_coeff
    #nmo = mf.mo_coeff.shape[1]
    #eri_mo = ao2mo.kernel(mol, c, intor='int2e_spinor').reshape(nmo,nmo,nmo,nmo)
    #hcore = mf.get_hcore()
    #h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
    #e = numpy.einsum('ij,ji', h1, rdm1)
    #e += numpy.einsum('ijkl,ijkl', eri_mo, rdm2)*0.5
    #e += mol.energy_nuc()
    #lib.logger.info(mf,"!*** E(MP2) with RDM: %s" % e)

