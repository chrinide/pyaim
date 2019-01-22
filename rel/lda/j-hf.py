#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import dhf
from pyscf.scf import _vhf
from pyscf.dft import gen_grid
from pyscf.dft import r_numint
from pyscf import __config__

def get_hcore(mol):
    '''2-component hcore Hamiltonian in the j-adapted spinor basis.'''
    if mol is None: mol = self.mol
    if mol.has_ecp():
        raise NotImplementedError
    t = mol.intor_symmetric('int1e_spsp_spinor') * .5
    v = mol.intor_symmetric('int1e_nuc_spinor')
    return t+v

def get_jk(mol, dm, hermi=1, mf_opt=None, with_j=True, with_k=True):
    '''J/K matrices in the j-adapted spinor basis.
    '''
    n2c = dm.shape[0]
    dd = numpy.zeros((n2c*2,)*2, dtype=numpy.complex128)
    dd[:n2c,:n2c] = dm
    dhf._call_veff_llll(mol, dd, hermi, None)
    vj, vk = _vhf.rdirect_mapdm('int2e_spinor', 's8',
                                ('ji->s2kl', 'jk->s1il'), dm, 1,
                                mol._atm, mol._bas, mol._env, mf_opt)
    return dhf._jk_triu_(vj, vk, hermi)

def make_rdm1(mo_coeff, mo_occ, **kwargs):
    return numpy.dot(mo_coeff*mo_occ, mo_coeff.T.conj())

def init_guess_by_minao(mol):
    '''Initial guess in terms of the overlap to minimal basis.'''
    dm = hf.init_guess_by_minao(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_1e(mol):
    '''Initial guess from one electron system.'''
    mf = UHF(mol)
    return mf.init_guess_by_1e(mol)

def init_guess_by_atom(mol):
    '''Initial guess from atom calculation.'''
    dm = hf.init_guess_by_atom(mol)
    return _proj_dmll(mol, dm, mol)

def init_guess_by_chkfile(mol, chkfile_name, project=None):
    dm = dhf.init_guess_by_chkfile(mol, chkfile_name, project)
    n2c = dm.shape[0] // 2
    return dm[:n2c,:n2c].copy()

def get_init_guess(mol, key='minao'):
    if callable(key):
        return key(mol)
    elif key.lower() == '1e':
        return init_guess_by_1e(mol)
    elif key.lower() == 'atom':
        return init_guess_by_atom(mol)
    elif key.lower() == 'chkfile':
        raise RuntimeError('Call pyscf.scf.hf.init_guess_by_chkfile instead')
    else:
        return init_guess_by_minao(mol)


class UHF(hf.SCF):
    def __init__(self, mol):
        hf.SCF.__init__(self, mol)

    def build(self, mol=None):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.opt = self.init_direct_scf(self.mol)

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        return self

    def init_guess_by_minao(self, mol=None):
        '''Initial guess in terms of the overlap to minimal basis.'''
        if mol is None: mol = self.mol
        return init_guess_by_minao(mol)

    def init_guess_by_atom(self, mol=None):
        if mol is None: mol = self.mol
        return init_guess_by_atom(mol)

    def init_guess_by_chkfile(self, chkfile=None, project=None):
        if chkfile is None: chkfile = self.chkfile
        return init_guess_by_chkfile(self.mol, chkfile, project=project)

    def _eigh(self, h, s):
        e, c = scipy.linalg.eigh(h, s)
        idx = numpy.argmax(abs(c.real), axis=0)
        c[:,c[idx,range(len(e))].real<0] *= -1
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def reset(self, mol):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        self.mol = mol
        return self

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('int1e_ovlp_spinor')

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:mol.nelectron] = 1
        if mol.nelectron < len(mo_energy):
            logger.info(self, 'nocc = %d  HOMO = %.12g  LUMO = %.12g', \
                        mol.nelectron, mo_energy[mol.nelectron-1],
                        mo_energy[mol.nelectron])
        else:
            logger.info(self, 'nocc = %d  HOMO = %.12g  no LUMO', \
                        mol.nelectron, mo_energy[mol.nelectron-1])
        logger.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return make_rdm1(mo_coeff, mo_occ, **kwargs)

    def init_direct_scf(self, mol=None):
        if mol is None: mol = self.mol
        def set_vkscreen(opt, name):
            opt._this.contents.r_vkscreen = _vhf._fpointer(name)
        opt = _vhf.VHFOpt(mol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                          'CVHFrkbllll_direct_scf',
                          'CVHFrkbllll_direct_scf_dm')
        opt.direct_scf_tol = self.direct_scf_tol
        set_vkscreen(opt, 'CVHFrkbllll_vkscreen')
        return opt

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        t0 = (time.clock(), time.time())
        if self.direct_scf and self.opt is None:
            self.opt = self.init_direct_scf(mol)
        vj, vk = get_jk(mol, dm, hermi, self.opt, with_j, with_k)
        logger.timer(self, 'vj and vk', *t0)
        return vj, vk

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.direct_scf:
            ddm = numpy.array(dm, copy=False) - numpy.array(dm_last, copy=False)
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return numpy.array(vhf_last, copy=False) + vj - vk
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return dhf.analyze(self, verbose)

try:
    from pyscf.dft import rks, dks
    class UKS(UHF):
        def __init__(self, mol):
            UHF.__init__(self, mol)
            self.xc = 'LDA,VWN'
            self.grids = gen_grid.Grids(self.mol)
            self.grids.level = getattr(__config__, 'dft_rks_RKS_grids_level',
                                       self.grids.level)
            # Use rho to filter grids
            self.small_rho_cutoff = getattr(__config__, 'dft_rks_RKS_small_rho_cutoff',
                                            1e-7)
    ##################################################
    # don't modify the following attributes, they are not input options
            self._numint = r_numint.RNumInt()
            self._keys = self._keys.union(['xc', 'grids', 'small_rho_cutoff'])
        def dump_flags(self):
            hf.SCF.dump_flags(self)
            logger.info(self, 'XC functionals = %s', self.xc)
            logger.info(self, 'small_rho_cutoff = %g', self.small_rho_cutoff)
            self.grids.dump_flags()
            return self

        get_veff = dks.get_veff
        energy_elec = rks.energy_elec
        define_xc_ = rks.define_xc_

except ImportError:
    pass

def _proj_dmll(mol_nr, dm_nr, mol):
    from pyscf.scf import addons
    proj = addons.project_mo_nr2r(mol_nr, numpy.eye(mol_nr.nao_nr()), mol)
    # *.5 because alpha and beta are summed in project_mo_nr2r
    dm_ll = reduce(numpy.dot, (proj, dm_nr*.5, proj.T.conj()))
    dm_ll = (dm_ll + dhf.time_reversal_matrix(mol, dm_ll)) * .5
    return dm_ll

# A tag to label the derived SCF class
class _X2C_SCF:
    pass


if __name__ == '__main__':
    from pyscf import scf
    mol = mole.Mole()
    mol.build(
        verbose = 4,
        spin = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    method = scf.UHF(mol)
    enr = method.kernel()
    print('E = %.12g' % enr)

    method = UHF(mol)
    ex2c = method.kernel()
    print('E j-basis = %.12g' % ex2c)

    method = UKS(mol)
    ex2c = method.kernel()
    print('E j-basis = %.12g' % ex2c)
