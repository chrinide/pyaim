#!/usr/bin/env python

import h5py
import numpy

xnuc = numpy.zeros(3)
xyzrho = numpy.zeros(3)

with h5py.File('surface.h5') as f:
    print('Surface file is a HDF5 file. data are stored in file/directory structure.')
    print('/', f.keys())
    print('/atom0', f['atom0'].keys())
    inuc = f['atom0/inuc'].value
    rmin = f['atom0/rmin'].value
    rmax = f['atom0/rmax'].value
    xyzrho = f['atom0/xyzrho'].value
    xnuc = f['atom0/xnuc'].value
    npang = f['atom0/npang'].value
    ntrial = f['atom0/ntrial'].value
    rsurf = numpy.zeros((npang,ntrial))
    nlimsurf = numpy.zeros((npang))
    coords = numpy.zeros((npang,4))
    rsurf = f['atom0/surface'].value
    nlimsurf = f['atom0/intersecs'].value
    coords = f['atom0/coords'].value

print 'inuc', inuc
print 'xnuc', xnuc
print 'xyzrho', xyzrho
print 'npang', npang
print 'ntrial', ntrial
print 'rmin', rmin
print 'rmax', rmax
for i in range(npang):
    #print coords[i,0],coords[i,1],coords[i,2],coords[i,3],coords[i,4],rsurf[i,0], nlimsurf[i]
    print '%16.8f %16.8f %16.8f %16.8f %16.8f %16.8f' % (coords[i,0],coords[i,1],coords[i,2],coords[i,3],coords[i,4],rsurf[i,0])
