#!/usr/bin/env python

import numpy, h5py
from pyscf import gto, scf, lib
from pyscf.tools.dump_mat import dump_tri

name = 'lih'
mol = lib.chkfile.load_mol(name+'.chk')
mo_coeff = scf.chkfile.load(name+'.chk', 'scf/mo_coeff')
mo_occ = scf.chkfile.load(name+'.chk', 'scf/mo_occ')
dm = scf.hf.make_rdm1(mo_coeff, mo_occ)

# Reference AO basis
with h5py.File(name+'_integrals.h5') as f:
    sref = f['molecule/overlap'].value
lib.logger.info(mol,'* REF Overlap on AO basis')
dump_tri(mol.stdout, sref, ncol=15, digits=5, start=0)

# Read info from aom program
nmo = mo_coeff.shape[1]
aom = numpy.zeros((mol.natm,nmo,nmo))
totaom = numpy.zeros((nmo,nmo))
with h5py.File(name+'.chk.h5') as f:
    for i in range(mol.natm):
        idx = 'ovlp'+str(i)
        aom[i] = f[idx+'/aom'].value
for i in range(mol.natm):
    totaom += aom[i]

# Transform total MO aom to AO basis and check diff
coeff = numpy.linalg.inv(mo_coeff)
lib.logger.info(mol,'* Total AOM on AO basis')
totaom = coeff.T.dot(totaom).dot(coeff)
dump_tri(mol.stdout, totaom, ncol=15, digits=5, start=0)
diff = numpy.linalg.norm(totaom-sref)
lib.logger.info(mol,'* Diff in AOM on %f' % diff)

