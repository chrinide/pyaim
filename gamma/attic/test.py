#!/usr/bin/env python

'''
Convert the k-sampled MO to corresponding Gamma-point supercell MO.
Zhihao Cui zcui@caltech.edu
'''

import numpy as np
import scipy
from scipy import linalg as la
import cmath, os, sys, copy

from pyscf import scf, gto, lo, lib, tools
from pyscf.lib import numpy_helper as np_helper
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc import df
from pyscf.pbc.tools import pbc as tools_pbc


def get_R_vec(mol, abs_kpts, kmesh):

    '''
    get supercell R vector based on k mesh
    '''

    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0]) 
    R_rel_b = np.arange(kmesh[1]) 
    R_rel_c = np.arange(kmesh[2])
    R_rel_mesh = np_helper.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_abs_mesh = np.einsum('nu, uv -> nv', R_rel_mesh, latt_vec)
    return R_abs_mesh

def nullspace(A, atol=5e-4, rtol=0): # NOTE: not used.

    '''
    Compute an approximate basis for the nullspace of A.
    http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    The algorithm used by this function is based on the singular value
    decomposition of `A`.
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    '''

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def find_degenerate(mo_energy, mo_coeff, real_split = False, tol = 1e-5):

    '''
    split the mo_energy into groups based on degenracy.
    further split the real MO out by set real_split to be True
    mo_energy should range from lowest to highest.
    return grouped mo_energy and its indices.    
    '''

    real_tol = tol * 0.01 # tol for split real
    
    res = []
    res_idx = []
    g_cur = [mo_energy[0]]
    idx_cur = [0]

    for i in xrange(1, len(mo_energy)):
        diff = mo_energy[i] - mo_energy[i-1]
        if diff < tol:
            g_cur.append(mo_energy[i])
            idx_cur.append(i)
        else:
            res.append(g_cur)
            res_idx.append(idx_cur)
            g_cur = [mo_energy[i]]
            idx_cur = [i]
        if i == len(mo_energy)-1:
            res.append(g_cur)
            res_idx.append(idx_cur)

    if real_split:
        res_idx_new = []
        res_new = []
        for i in xrange(len(res_idx)):
            res_idx_tmp = copy.deepcopy(res_idx[i])
            res_tmp = copy.deepcopy(res[i])
            
            tmp = 0            

            for j in xrange(len(res_idx_tmp)):
                if la.norm(mo_coeff[:,res_idx_tmp[j-tmp]].imag) < real_tol:
                    p = res_idx_tmp.pop(j-tmp)
                    res_idx_new.append([p])

                    e = res_tmp.pop(j-tmp)
                    res_new.append([e])

                    tmp += 1

            if res_idx_tmp != []:
                    res_idx_new.append(res_idx_tmp) 
                    res_new.append(res_tmp)
                
        # sort again to make sure the slightly lower energy state to be the first
        sort_idx = sorted(range(len(res_new)), key=lambda k: res_new[k])
        res_new = [res_new[i] for i in sort_idx]
        res_idx_new = [res_idx_new[i] for i in sort_idx]
        res_idx = res_idx_new
        res = res_new

    return res, res_idx


def k2gamma(kmf, abs_kpts, kmesh, realize = True, real_split = False, tol_deg = 5e-5):

    '''
    convert the k-sampled mo coefficient to corresponding supercell gamma-point mo coefficient.
    set realize = True to make sure the final wavefunction to be real.
    return the supercell gamma point object 
    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \qty[ \frac{1}{\sqrt{N_{\UC}}} \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}]
    '''

    #np.set_printoptions(4,linewidth=1000)

    R_abs_mesh = get_R_vec(kmf.cell, abs_kpts, kmesh)    
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk',R_abs_mesh, abs_kpts))
    
    E_k = np.asarray(kmf.mo_energy)
    occ_k = np.asarray(kmf.mo_occ)
    C_k = np.asarray(kmf.mo_coeff)
    
    Nk, Nao, Nmo = C_k.shape
    NR = R_abs_mesh.shape[0]    

    C_gamma = np.einsum('Rk, kum -> Rukm', phase, C_k) / np.sqrt(NR)
    C_gamma = C_gamma.reshape((NR, Nao, Nk*Nmo))

    # sort energy of km
    E_k_flat = E_k.flatten()
    E_k_sort_idx = np.argsort(E_k_flat)
    E_k_sort = E_k_flat[E_k_sort_idx]
    occ_sort = occ_k.flatten()[E_k_sort_idx]
    
    C_gamma = C_gamma[:, :, E_k_sort_idx]
    C_gamma = C_gamma.reshape((NR*Nao, Nk*Nmo))

    # supercell object
    sc = tools_pbc.super_cell(cell, kmesh)
    sc.verbose = 0
    kmf_sc = pscf.KRHF(sc, [[0.0,0.0,0.0]]).density_fit() # TODO spin polarize ? DF or not?
    S_sc = kmf_sc.get_ovlp()[0].real
   
    # make MO to be real
    if realize:
        
        real_tol = tol_deg # tolerance of residue of real or imag part
        null_tol = min(tol_deg * 10.0, 1.0e-3) # tolerance of 0 for nat_orb selection 

        print "Realize the gamma point MO ..." 
        C_gamma_real = np.zeros_like(C_gamma, dtype = np.double)

        res, res_idx = find_degenerate(E_k_sort, C_gamma, real_split = real_split, tol = tol_deg )
        print "Energy spectrum group:", res
        print "Energy idx:", res_idx
        col_idx = 0
        for i, gi_idx in enumerate(res_idx):
            gi = C_gamma[:,gi_idx]

            # using dm to solve natural orbitals, to make the orbitals real
            dm =  gi.dot(gi.conj().T)
            if la.norm(dm.imag) > real_tol:
                print "density matrix of converted Gamma MO has large imaginary part."
                sys.exit(1)
            eigval, eigvec = la.eigh(dm.real, S_sc, type = 2)
            nat_orb = eigvec[:, eigval > null_tol]
            if nat_orb.shape[1] != len(gi_idx):
                print "Realization error, not find correct number of linear combination coefficient"
                sys.exit(1)
            for j in xrange(nat_orb.shape[1]):
                C_gamma_real[:,col_idx] = nat_orb[:, j]
                col_idx += 1

        C_gamma = C_gamma_real
    
    # save to kmf_sc obj
    kmf_sc.mo_coeff = [C_gamma]
    kmf_sc.mo_energy = [np.asarray(E_k_sort)]
    kmf_sc.mo_occ = [np.asarray(occ_sort)]

    return kmf_sc

            
if __name__ == '__main__':
    
    np.set_printoptions(3,linewidth=1000)

    cell = pgto.Cell()
    cell.atom = '''C     0.      0.      0.    
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.    
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751'''
    cell.a = np.eye(3)*3.5668
    cell.basis = 'sto-3g'
    #cell.precision = 1e-10
    cell.verbose = 4
    cell.build()
    
    kmesh = [2, 2, 2]
    abs_kpts = cell.make_kpts(kmesh)
    scaled_kpts = cell.get_scaled_kpts(abs_kpts)

    kmf = pscf.KRHF(cell, abs_kpts).density_fit()
    gdf = df.GDF(cell, abs_kpts)
    kmf.with_df = gdf
    #kmf.checkfile = './ch4.chk'
    kmf.verbose = 5
    #kmf = pscf.KRHF(cell, abs_kpts)
    #kmf.__dict__.update(scf.chkfile.load('ch4.chk', 'scf')) # test
    ekpt = kmf.run()

    print np.asarray(kmf.mo_coeff)

    kmf_sc = k2gamma(kmf, abs_kpts, kmesh, realize = True, tol_deg = 5e-5, real_split = False)
    c_g_ao = kmf_sc.mo_coeff[0] 
    print "Supercell gamma MO in AO basis from conversion:"
    print c_g_ao

    dm_convert = 2.0 * c_g_ao[:, kmf_sc.mo_occ[0]>0].dot(c_g_ao[:, kmf_sc.mo_occ[0]>0].conj().T)


    # The following is to check whether the MO is correctly coverted: 

    sc = tools_pbc.super_cell(cell, kmesh)
    #sc.verbose = 0


    kmf_sc2 = pscf.KRHF(sc, [[0.0, 0.0, 0.0]]).density_fit()
    gdf = df.GDF(sc, [[0.0, 0.0, 0.0]])
    kmf_sc2.with_df = gdf
    s = kmf_sc2.get_ovlp()[0]

    print "Run supercell gamma point calculation..." 
    ekpt_sc = kmf_sc2.run([dm_convert])
    #ekpt_sc = kmf_sc2.run()
    sc_mo = kmf_sc2.mo_coeff[0]
    print "Supercell gamma MO from direct calculation:"
    print sc_mo

    # lowdin of sc_mo and c_g_ao
    sc_mo_o = la.sqrtm(s).dot(sc_mo)
    c_g = la.sqrtm(s).dot(c_g_ao)

    res, res_idx = find_degenerate(kmf_sc.mo_energy[0], c_g, real_split = False, tol = 5e-5)

    print res
    print kmf_sc2.mo_energy
    print c_g_ao.conj().T.dot(s).dot(c_g_ao)
    exit()

    for i in xrange(len(res_idx)): 
        print 
        print "subspace:", i
        print "index:", res_idx[i]
        print "energy:", res[i]
        #print sc_mo.conj().T.dot(s.dot(sc_mo)).real
        u, sigma, v = la.svd(c_g[:,res_idx[i]].T.conj().dot(sc_mo_o[:,res_idx[i]]))
        print "singular value of subspace (C_convert * C_calculated):" , sigma
        


    #name = 'k2gamma'
    #kpts = [0,0,0]
    #lib.chkfile.save_cell(name+'.chk', scell)
    #lib.chkfile.save(name+'.chk', 'kcell/kpts', kpts)
    #lib.chkfile.save(name+'.chk', 'scf/mo_coeff', mf.mo_coeff)
    #lib.chkfile.save(name+'.chk', 'scf/mo_occ', mf.mo_occ)
