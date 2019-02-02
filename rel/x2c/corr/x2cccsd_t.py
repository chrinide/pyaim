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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import numpy
from pyscf import lib
from pyscf.lib import logger

# spin-orbital formula
# JCP, 98, 8718
def kernel(cc, eris, t1=None, t2=None, verbose=logger.INFO):
    if t1 is None or t2 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nvir = t1.shape

    bcei = numpy.asarray(eris.ovvv).conj().transpose(3,2,1,0)
    majk = numpy.asarray(eris.ooov).conj().transpose(2,3,0,1)
    bcjk = numpy.asarray(eris.oovv).conj().transpose(2,3,0,1)
    fvo = eris.fock[nocc:,:nocc]
    mo_e = eris.mo_energy
    eijk = lib.direct_sum('i+j+k->ijk', mo_e[:nocc], mo_e[:nocc], mo_e[:nocc])
    eabc = lib.direct_sum('a+b+c->abc', mo_e[nocc:], mo_e[nocc:], mo_e[nocc:])

    t2T = t2.transpose(2,3,0,1)
    t1T = t1.T
    def get_wv(a, b, c):
        w  = lib.einsum('ejk,ei->ijk', t2T[a,:], bcei[b,c])
        w -= lib.einsum('im,mjk->ijk', t2T[b,c], majk[:,a])
        v  = lib.einsum('i,jk->ijk', t1T[a], bcjk[b,c])
        v += lib.einsum('i,jk->ijk', fvo[a], t2T [b,c])
        v += w
        w = w + w.transpose(2,0,1) + w.transpose(1,2,0)
        return w, v

    et = 0
    for a in range(nvir):
        for b in range(a):
            for c in range(b):
                wabc, vabc = get_wv(a, b, c)
                wcab, vcab = get_wv(c, a, b)
                wbac, vbac = get_wv(b, a, c)

                w = wabc + wcab - wbac
                v = vabc + vcab - vbac
                w /= eijk - eabc[a,b,c]
                et += numpy.einsum('ijk,ijk', w, v.conj())
    et /= 2
    return et


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
    
    mf = x2c.UHF(mol)
    dm = mf.get_init_guess() + 0.1j
    mf.kernel(dm)

    ncore = 2
    import x2cccsd
    mycc = x2cccsd.GCCSD(mf)
    mycc.frozen = ncore
    mycc.kernel()
    eris = mycc.ao2mo()
    print(kernel(mycc, eris))

