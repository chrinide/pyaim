#!/usr/bin/env python

import sys
sys.path.append('..')
import basin

natm = 6
name = 'c2f4.chk'
bas = basin.Basin(name)
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
for i in range(natm):
    bas.inuc = i
    bas.kernel()

