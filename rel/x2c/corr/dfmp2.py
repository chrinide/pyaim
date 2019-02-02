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
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf import df
from pyscf.mp import mp2
from pyscf import scf, x2c
from pyscf import __config__

WITH_T2 = getattr(__config__, 'x2c_mp2_with_t2', True)
THRESH_VIR = getattr(__config__, 'x2c_mp2_fno_thresh_vir', 1e-4)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mo_energy is None:
        mo_energy = mp.mo_energy[mp.get_frozen_mask()]
    else:
        mo_energy = mo_energy[mp.get_frozen_mask()]

    if eris is None: eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eo = mo_energy[:nocc] 
    ev = mo_energy[nocc:]

    if with_t2:
        t2 = numpy.zeros((nocc,nvir,nocc,nvir), dtype=numpy.complex128)
    else:
        t2 = None

    emp2 = 0.0
    vv_denom = -ev.reshape(-1,1)-ev
    for i in range(nocc):
    	eps_i = eo[i]
    	i_Qv = eris.ov[:, i, :].copy()
    	for j in range(nocc):
            eps_j = eo[j]
            j_Qv = eris.ov[:, j, :].copy()
            viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
            vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
            v = viajb - vibja
            div = 1.0/(eps_i + eps_j + vv_denom)
            tmp = v.conj()*div 
            emp2 += 0.25*numpy.einsum('ab,ab->', v, tmp) 
            if with_t2:
                t2[i,:,j,:] += tmp

    if with_t2:
        t2 = t2.transpose(0,2,1,3)
    return emp2.real, t2

def get_fno(mp, mo_energy=None, mo_coeff=None, eris=None, thresh_vir=THRESH_VIR, 
            verbose=logger.NOTE):

    lib.logger.info(mp,"\n* Fno procedure")
    lib.logger.info(mp,"* Threshold for virtual occupation %g", thresh_vir)

    if mo_energy is None:
        mo_energy = mp.mo_energy[mp.get_frozen_mask()]
    else:
        mo_energy = mo_energy[mp.get_frozen_mask()]

    if eris is None: eris = mp.ao2mo(mo_coeff)

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eo = mo_energy[:nocc] 
    ev = mo_energy[nocc:]

    dab = numpy.zeros((len(ev),len(ev)), dtype=numpy.complex128)
    vv_denom = -ev.reshape(-1,1)-ev
    for i in range(nocc):
        eps_i = eo[i]
        i_Qv = eris.ov[:, i, :].copy()
        for j in range(nocc):
            eps_j = eo[j]
            j_Qv = eris.ov[:, j, :].copy()
            viajb = lib.einsum('Qa,Qb->ab', i_Qv, j_Qv)
            vibja = lib.einsum('Qb,Qa->ab', i_Qv, j_Qv)
            v = viajb - vibja
            div = 1.0/(eps_i + eps_j + vv_denom)
            t2ij = v.conj()*div
            dab += lib.einsum('ea,eb->ab', t2ij,t2ij.conj())*0.5

    dm1 = dab + dab.conj().T
    dm1 *= 0.5

    natoccvir, natorbvir = numpy.linalg.eigh(-dm1)
    for i, k in enumerate(numpy.argmax(abs(natorbvir), axis=0)):
        if natorbvir[k,i] < 0:
            natorbvir[:,i] *= -1
    natoccvir = -natoccvir
    lib.logger.debug(mp,"* Occupancies")
    lib.logger.debug(mp,"* %s" % natoccvir)
    lib.logger.debug(mp,"* The sum is %8.6f" % numpy.sum(natoccvir)) 
    active = (thresh_vir <= natoccvir)
    lib.logger.info(mp,"* Natural Orbital selection")
    for i in range(nvir):
        lib.logger.debug(mp,"orb: %d %s %8.6f" % (i,active[i],natoccvir[i]))
        actIndices = numpy.where(active)[0]
    lib.logger.info(mp,"* Original active orbitals %d" % nvir)
    lib.logger.info(mp,"* Virtual core orbitals: %d" % (nvir-len(actIndices)))
    lib.logger.info(mp,"* New active orbitals %d" % len(actIndices))
    lib.logger.debug(mp,"* Active orbital indices %s" % actIndices)
    natorbvir = natorbvir[:,actIndices]                                    
    ev = mo_energy[nocc:]
    fvv = numpy.diag(ev)
    fvv = reduce(numpy.dot, (natorbvir.conj().T, fvv, natorbvir))
    ev, fnov = numpy.linalg.eigh(fvv)
    cv = mp.mo_coeff[:,mp.get_frozen_mask()]
    cv = cv[:,nocc:]
    cv = reduce(numpy.dot, (cv, natorbvir, fnov))
    co = mp.mo_coeff[:,mp.mo_occ>0]
    eo = mp.mo_energy[mp.mo_occ>0]
    coeff = numpy.hstack([co,cv])
    energy = numpy.hstack([eo,ev]) 
    occ = numpy.zeros(coeff.shape[1])
    for i in range(mp.mol.nelectron):
        occ[i] = 1.0
    return coeff, energy, occ

class DFMP2(x2c.mp2.MP2):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
        if getattr(mf, 'with_df', None):
            self.with_df = mf.with_df
        else:
            self.with_df = df.DF(mf.mol)
            self.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
            sefl.with_df.build()
        self._keys.update(['with_df'])

    @lib.with_doc(mp2.MP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return mp2.MP2.kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def fno(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return get_fno(self, mo_energy, mo_coeff, eris)
    
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = nocc**2*nvir**2*4 * 8/1e6
        mem_now = lib.current_memory()[0]
        if ((mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff, verbose=self.verbose)
        else:
            raise NotImplementedError

MP2 = DFMP2

class _ChemistERIs:
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        self.mp = mp
        self.nocc = self.mp.nocc
        mo_idx = mp.get_frozen_mask()
        self.mo_coeff = mo_coeff[:,mo_idx]
        self.ov = None

def _make_eris_incore(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    eris = _ChemistERIs(mp, mo_coeff)
    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    orbo = eris.mo_coeff[:,:nocc]
    orbv = eris.mo_coeff[:,nocc:]
    dferi = mp.with_df._cderi.reshape(-1,nao,nao)
    eris.ov = lib.einsum('rj,Qrs->Qjs', orbo.conj(), dferi)
    eris.ov = lib.einsum('sb,Qjs->Qjb', orbv, eris.ov)
    return eris

del(WITH_T2)

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
    
    mf = x2c.RHF(mol).density_fit()
    dm = mf.get_init_guess() + 0.0j
    mf.with_df.auxbasis = 'def2-svp-jkfit'
    mf.kernel(dm)

    ncore = 2

    pt = MP2(mf)
    pt.frozen = ncore
    pt.kernel()
    rdm1 = pt.make_rdm1()
    rdm2 = pt.make_rdm2()

    n2c = mol.nao_2c()
    dferi = mf.with_df._cderi.reshape(-1,n2c,n2c)
    eri_mo = lib.einsum('rj,Qrs->Qjs', mf.mo_coeff.conj(), dferi)
    eri_mo = lib.einsum('sb,Qjs->Qjb', mf.mo_coeff, eri_mo)
    eri_mo = lib.einsum('Qia,Qjb->iajb', eri_mo, eri_mo)
    hcore = mf.get_hcore()
    h1 = reduce(numpy.dot, (mf.mo_coeff.T.conj(), hcore, mf.mo_coeff))
    e = numpy.einsum('ij,ji', h1, rdm1)
    e += numpy.einsum('ijkl,ijkl', eri_mo, rdm2)*0.5
    e += mol.energy_nuc()
    print("!*** E(X2CMP2) with RDM: %s" % e)

    mo_coeff, mo_energy, mo_occ = pt.fno()

    pt = MP2(mf, mo_coeff=mo_coeff, mo_occ=mo_occ)
    pt.frozen = ncore
    pt.kernel(mo_energy=mo_energy)
