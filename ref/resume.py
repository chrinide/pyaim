#!/usr/bin/env python

import sys
sys.path.append('..')
import tools

natm = 3
name = 'h2o.chk.h5'
for i in range(natm):
    tools.print_surface_txt(name, i) 

