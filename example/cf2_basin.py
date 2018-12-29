#!/usr/bin/env python

import basin

name = 'cf2.chk'
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

bas.inuc = 0
bas.kernel()

bas.inuc = 1
bas.kernel()

bas.inuc = 2
bas.kernel()
