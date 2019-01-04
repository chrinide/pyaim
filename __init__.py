#!/usr/bin/env python

__version__ = '0.1'

import numpy
from distutils.version import LooseVersion

if LooseVersion(numpy.__version__) <= LooseVersion('1.8.0'):
    raise SystemError("You're using an old version of Numpy (%s). "
                      "It is recommended to upgrad numpy to 1.8.0 or newer. \n"
                      "You still can use all features of PySCF with the old numpy by removing this warning msg. "
                      "Some modules (DFT, CC, MRPT) might be affected because of the bug in old numpy." %
                      numpy.__version__)

import __config__

DEBUG = __config__.DEBUG

del(LooseVersion, numpy)

