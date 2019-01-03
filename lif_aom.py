#!/usr/bin/env python

import aom

name = 'lif.chk'
bas = aom.Aom(name)
bas.verbose = 4
bas.nrad = 101
bas.iqudr = 'legendre'
bas.mapr = 'exp'
bas.bnrad = 101
bas.bnpang = 5810
bas.biqudr = 'legendre'
bas.bmapr = 'exp'
bas.non0tab = False
bas.full = False

bas.inuc = 0
bas.kernel()

bas.inuc = 1
bas.kernel()

