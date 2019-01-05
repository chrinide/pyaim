#!/usr/bin/env python

import sys
sys.path.append('..')
import aom

natm = 6
name = 'c2f4.chk'
ovlp = aom.Aom(name)
ovlp.verbose = 4
ovlp.nrad = 321
ovlp.iqudr = 'legendre'
ovlp.mapr = 'exp'
ovlp.bnrad = 221
ovlp.bnpang = 3074
ovlp.biqudr = 'legendre'
ovlp.bmapr = 'exp'
ovlp.betafac = 0.4
ovlp.corr = False
ovlp.occdrop = 1e-6
ovlp.full = False
for i in range(natm):
    ovlp.inuc = i
    ovlp.kernel()

