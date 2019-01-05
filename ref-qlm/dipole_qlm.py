#!/usr/bin/env python

import sys
sys.path.append('..')
import qlm

natm = 2
name = 'co.chk'
bas = qlm.Qlm(name)
bas.verbose = 4
bas.nrad = 321
bas.iqudr = 'legendre'
bas.mapr = 'exp'
bas.bnrad = 221
bas.bnpang = 3074
bas.biqudr = 'legendre'
bas.bmapr = 'exp'
bas.betafac = 0.4
bas.corr = False
bas.occdrop = 1e-6
bas.lmax = 10
bas.blmax = 10
for i in range(natm):
    bas.inuc = i
    bas.kernel()

