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

'''
GMP2 in spin-orbital form
E(MP2) = 1/4 <ij||ab><ab||ij>/(ei+ej-ea-eb)
'''

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.mp import mp2
from pyscf import scf
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_gmp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is None:
        mo_energy = mp.mo_energy[mp.get_frozen_mask()]
    else:
        mo_energy = mo_energy[mp.get_frozen_mask()]

    mo_e = mo_energy
    gap = abs(mo_e[:mp.nocc,None] - mo_e[None,mp.nocc:]).min()
    if gap < 1e-5:
        logger.warn(mp, 'HOMO-LUMO gap %s too small', gap)
    else:
        logger.info(mp, 'HOMO-LUMO gap %s', gap)

    if eris is None: eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    moidx = mp.get_frozen_mask()
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=numpy.complex128)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        gi = numpy.asarray(eris.oovv[i]).reshape(nocc,nvir,nvir)
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * .25
        if with_t2:
            t2[i] = t2i

    return emp2.real, t2

def make_rdm1(mp, t2=None, ao_repr=False):
    r'''
    One-particle density matrix in the molecular spin-orbital representation
    (the occupied-virtual blocks from the orbital response contribution are
    not included).

    dm1[p,q] = <q^\dagger p>  (p,q are spin-orbitals)

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)
    '''
    from pyscf.cc import gccsd_rdm
    if t2 is None: t2 = mp.t2
    doo, dvv = _gamma1_intermediates(mp, t2)
    nocc, nvir = t2.shape[1:3]
    dov = numpy.zeros((nocc,nvir))
    d1 = doo, dov, dov.T, dvv
    return gccsd_rdm._make_rdm1(mp, d1, with_frozen=True, ao_repr=ao_repr)

def make_rdm1_vv(mp, t2=None):
    if t2 is None: t2 = mp.t2
    dvv = lib.einsum('mnea,mneb->ab', t2, t2.conj()) * .5
    dm1 = dvv + dvv.conj().T
    dm1 *= .5
    return dm1

def _gamma1_intermediates(mp, t2):
    doo = lib.einsum('imef,jmef->ij', t2.conj(), t2) *-.5
    dvv = lib.einsum('mnea,mneb->ab', t2, t2.conj()) * .5
    return doo, dvv

# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mp, t2=None):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if t2 is None: t2 = mp.t2
    nmo = nmo0 = mp.nmo
    nocc = nocc0 = mp.nocc

    if not (mp.frozen is 0 or mp.frozen is None):
        nmo0 = mp.mo_occ.size
        nocc0 = numpy.count_nonzero(mp.mo_occ > 0)
        moidx = mp.get_frozen_mask()
        oidx = numpy.where(moidx & (mp.mo_occ > 0))[0]
        vidx = numpy.where(moidx & (mp.mo_occ ==0))[0]

        dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=t2.dtype) # Chemist's notation
        dm2[oidx[:,None,None,None],vidx[:,None,None],oidx[:,None],vidx] = \
                t2.transpose(0,2,1,3)
        dm2[nocc0:,:nocc0,nocc0:,:nocc0] = \
                dm2[:nocc0,nocc0:,:nocc0,nocc0:].transpose(1,0,3,2).conj()
    else:
        dm2 = numpy.zeros((nmo0,nmo0,nmo0,nmo0), dtype=t2.dtype) # Chemist's notation
        dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,2,1,3)
        dm2[nocc:,:nocc,nocc:,:nocc] = dm2[:nocc,nocc:,:nocc,nocc:].transpose(1,0,3,2).conj()

    dm1 = make_rdm1(mp, t2)
    dm1[numpy.diag_indices(nocc0)] -= 1

    # Be careful with convention of dm1 and dm2
    #   dm1[q,p] = <p^\dagger q>
    #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
    #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
    # When adding dm1 contribution, dm1 subscripts need to be flipped
    for i in range(nocc0):
        dm2[i,i,:,:] += dm1.T
        dm2[:,:,i,i] += dm1.T
        dm2[:,i,i,:] -= dm1.T
        dm2[i,:,:,i] -= dm1

    for i in range(nocc0):
        for j in range(nocc0):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= 1

    return dm2


class GMP2(mp2.MP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.incore = False
        self._keys = self._keys.union(['incore'])

    @lib.with_doc(mp2.MP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return mp2.MP2.kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        #mem_incore = nocc**2*nvir**2*3 * 8/1e6
        mem_now = lib.current_memory()[0]
        #if (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
        if (self.incore):
            logger.info(self,'Incore mem for 2e integrals')
            return _make_eris_incore(self, mo_coeff, verbose=self.verbose)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            logger.info(self,'Outcore mem for 2e integrals')
            return _make_eris_outcore(self, mo_coeff, self.verbose)

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2
    make_rdm1_vv = make_rdm1_vv


class _PhysicistsERIs:
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        self.mp = mp
        mo_idx = mp.get_frozen_mask()
        self.mo_coeff = mo_coeff[:,mo_idx]
        self.oovv = None
        self.feri = None

def _make_eris_incore(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    eris = _PhysicistsERIs(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc

    if callable(ao2mofn):
        orbo = eris.mo_coeff[:,:nocc]
        orbv = eris.mo_coeff[:,nocc:]
        eri = ao2mofn((orbo,orbv,orbo,orbv)).reshape(nocc,nvir,nocc,nvir)
    else:
        orbo = eris.mo_coeff[:,:nocc]
        orbv = eris.mo_coeff[:,nocc:]
        eri = ao2mo.kernel(mp.mol.intor('int2e_spinor'), (orbo,orbv,orbo,orbv)).reshape(nocc,nvir,nocc,nvir)

    eris.oovv = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    return eris

def _make_eris_outcore(mp, mo_coeff=None, verbose=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mp.stdout, mp.verbose)
    eris = _PhysicistsERIs(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    orbo = eris.mo_coeff[:,:nocc]
    orbv = eris.mo_coeff[:,nocc:]

    fswap = eris.feri = lib.H5TmpFile()
    eris.oovv = fswap.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'c8')

    max_memory = mp.max_memory-lib.current_memory()[0]
    blksize = min(nocc, max(2, int(max_memory*1e6/8/(nocc*nvir**2*2))))
    max_memory = max(2000, max_memory)

    ao2mo.kernel(mp.mol, (orbo,orbv,orbo,orbv), fswap,
                 max_memory=max_memory, verbose=log, intor='int2e_spinor')

    for p0, p1 in lib.prange(0, nocc, blksize):
        tmp = numpy.asarray(fswap['eri_mo'][p0*nvir:p1*nvir])
        tmp = tmp.reshape(p1-p0,nvir,nocc,nvir)
        eris.oovv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1) 
        tmp = None
    cput0 = log.timer_debug1('transforming oovv', *cput0)

    return eris

del(WITH_T2)

