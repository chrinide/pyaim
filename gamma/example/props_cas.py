#!/usr/bin/env python

from pyaim.gamma import basin

name = 'gamma_cas.chk'
natm = 2

props = basin.Basin(name)
props.verbose = 4
props.nrad = 221
props.iqudr = 'legendre'
props.mapr = 'becke'
props.bnrad = 121
props.bnpang = 3074
props.biqudr = 'legendre'
props.bmapr = 'exp'
props.betafac = 0.5
props.corr = True
props.cas = True
for i in range(natm):
    props.inuc = i
    props.kernel()

