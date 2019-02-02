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
from pyscf.x2c import x2c
from pyscf.x2c import ccsd
from pyscf.x2c import ccsd_t
from pyscf.x2c import mp2
from pyscf.x2c import dfmp2
from pyscf.x2c import cisd

def RHF(mol, *args):
    return x2c.RHF(mol, *args)

def UHF(mol, *args):
    return x2c.UHF(mol, *args)

def RKS(mol, *args):
    return x2c.RKS(mol, *args)

def UKS(mol, *args):
    return x2c.UKS(mol, *args)

def MP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if getattr(mf, 'with_df', None):
        return dfmp2.MP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.MP2(mf, frozen, mo_coeff, mo_occ)

def CCSD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if getattr(mf, 'with_df', None):
        raise NotImplementedError
    else:
        return ccsd.CCSD(mf, frozen, mo_coeff, mo_occ)

def CISD(mf, frozen=0, mo_coeff=None, mo_occ=None):
    if getattr(mf, 'with_df', None):
        raise NotImplementedError
    else:
        return cisd.CISD(mf, frozen, mo_coeff, mo_occ)

