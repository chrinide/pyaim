#!/usr/bin/env python

import basin

name = 'crco6.chk'
bas = basin.Basin(name)
bas.verbose = 4
bas.nrad = 101
bas.iqudr = 'legendre'
bas.mapr = 'becke'
bas.bnrad = 101
bas.bnpang = 5810
bas.biqudr = 'legendre'
bas.bmapr = 'becke'
bas.non0tab = False

natm = 13
for i in range(natm):
    bas.inuc = i
    bas.kernel()

