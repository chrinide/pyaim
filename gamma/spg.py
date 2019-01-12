#!/usr/bin/env python

import os
import sys
import time
import numpy
import ctypes
import signal

try:
    import spglib as spg
except ImportError:
    from pyspglib import spglib as spg

from pyscf.pbc import lib as libpbc
from pyscf import lib
from pyscf.lib import logger

signal.signal(signal.SIGINT, signal.SIG_DFL)

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

# cell
# | ax  ay  az |
# | bx  by  bz |
# | cx  cy  cz |
def frac2cart(cell, coords):
    natm = coords.shape[0]
    cart = numpy.zeros((natm,3))
    for i in range(natm):
        cart[i,0] = coords[i,0]*cell[0,0] + coords[i,1]*cell[1,0] + coords[i,2]*cell[2,0]
        cart[i,1] = coords[i,0]*cell[0,1] + coords[i,1]*cell[1,1] + coords[i,2]*cell[2,1]
        cart[i,2] = coords[i,0]*cell[0,2] + coords[i,1]*cell[1,2] + coords[i,2]*cell[2,2]
    return cart

# Uses Cramer's Rule to solve for the fractional coordinates
# cell
# | ax  ay  az |
# | bx  by  bz |
# | cx  cy  cz |
# use the transpose of this matrix
def cart2frac(cell, coords):
  natm = coords.shape[0]
  frac = numpy.zeros((natm,3))
  det = numpy.linalg.det(cell)
  def det3(a):
    return numpy.linalg.det(a)
  tmp = numpy.zeros((3,3))
  for i in range(natm):
    tmp[0] = [coords[i,0], cell[1,0], cell[2,0]]
    tmp[1] = [coords[i,1], cell[1,1], cell[2,1]]
    tmp[2] = [coords[i,2], cell[1,2], cell[2,2]]
    aPos = det3(tmp) / det                      
    tmp[0] = [cell[0,0], coords[i,0], cell[2,0]]
    tmp[1] = [cell[0,1], coords[i,1], cell[2,1]]
    tmp[2] = [cell[0,2], coords[i,2], cell[2,2]]
    bPos = det3(tmp) / det                      
    tmp[0] = [cell[0,0], cell[1,0], coords[i,0]]
    tmp[1] = [cell[0,1], cell[1,1], coords[i,1]]
    tmp[2] = [cell[0,2], cell[1,2], coords[i,2]]
    cPos = det3(tmp) / det                      
    frac[i,0] = aPos
    frac[i,1] = bPos
    frac[i,2] = cPos
  return frac

def show_symmetry(symmetry):
    for i in range(symmetry['rotations'].shape[0]):
        print("  --------------- %4d ---------------" % (i + 1))
        rot = symmetry['rotations'][i]
        trans = symmetry['translations'][i]
        print("  rotation:")
        for x in rot:
            print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
        print("  translation:")
        print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))

def show_lattice(lattice):
    print("Basis vectors:")
    for vec, axis in zip(lattice, ("a", "b", "c")):
        print("%s %10.5f %10.5f %10.5f" % (tuple(axis,) + tuple(vec)))

def show_cell(lattice, positions, numbers):
    show_lattice(lattice)
    print("Atomic points:")
    for p, s in zip(positions, numbers):
        print("%2d %10.5f %10.5f %10.5f" % ((s,) + tuple(p)))

def spginfo(self):
    cell = (self.a,self.frac,self.charges)
    print("[get_spacegroup]")
    print("  Spacegroup of cell is %s" % spg.get_spacegroup(cell))
    print("[get_symmetry]")
    print("  Symmetry operations of unitcell are")
    symmetry = spg.get_symmetry(cell)
    show_symmetry(symmetry)
    print("[get_pointgroup]")
    print("  Pointgroup of cell is %s" %
    spg.get_pointgroup(symmetry['rotations'])[0])
    dataset = spg.get_symmetry_dataset(cell)
    print("[get_symmetry_dataset] ['international']")
    print("  Spacegroup of cell is %s (%d)" % (dataset['international'],
                                               dataset['number']))
    print("[get_symmetry_dataset] ['pointgroup']")
    print("  Pointgroup of cell is %s" % (dataset['pointgroup']))
    print("[get_symmetry_dataset] ['hall']")
    print("  Hall symbol of cell is %s (%d)." % (dataset['hall'],
                                                 dataset['hall_number']))
    print("[get_symmetry_dataset] ['wyckoffs']")
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    print("  Wyckoff letters of cell are ", dataset['wyckoffs'])
    print("[get_symmetry_dataset] ['equivalent_atoms']")
    print("  Mapping to equivalent atoms of cell ")
    for i, x in enumerate(dataset['equivalent_atoms']):
    	print("  %d -> %d" % (i+1, x+1))
    print("[get_symmetry_dataset] ['rotations'], ['translations']")
    print("  Symmetry operations of unitcell are:")
    for i, (rot,trans) in enumerate(zip(dataset['rotations'],
                                        dataset['translations'])):
        print("  --------------- %4d ---------------" % (i+1))
    print("  rotation:")
    for x in rot:
        print("     [%2d %2d %2d]" % (x[0], x[1], x[2]))
    print("  translation:")
    print("     (%8.5f %8.5f %8.5f)" % (trans[0], trans[1], trans[2]))
    reduced_lattice = spg.niggli_reduce(self.a)
    print("[niggli_reduce]")
    print("  Original lattice")
    show_lattice(self.a)
    print("  Reduced lattice")
    show_lattice(reduced_lattice)
    mapping, grid = spg.get_ir_reciprocal_mesh([11, 11, 11],
                                                cell,
                                                is_shift=[0, 0, 0])
    num_ir_kpt = len(numpy.unique(mapping))
    print("[get_ir_reciprocal_mesh]")
    print("  Number of irreducible k-points of primitive")
    print("  11x11x11 Monkhorst-Pack mesh is %d " % num_ir_kpt)

    return self

class Spg(lib.StreamObject):

    def __init__(self, datafile):
        self.verbose = logger.NOTE
        self.stdout = sys.stdout
        self.max_memory = lib.param.MAX_MEMORY
        self.chkfile = datafile
        self.surfile = datafile+'.h5'
        self.scratch = lib.param.TMPDIR 
        self.nthreads = lib.num_threads()
##################################################
# don't modify the following attributes, they are not input options
        self.cell = None
        self.vol = None
        self.a = None
        self.b = None
        self.kpts = None
        self.nkpts = None
        self.ls = None
        self.natm = None
        self.coords = None
        self.charges = None
        self.nelectron = None
        self.charge = None
        self.spin = None
        self.frac = None
        self._keys = set(self.__dict__.keys())

    def dump_input(self):

        if self.verbose < logger.INFO:
            return self

        logger.info(self,'')
        logger.info(self,'******** %s flags ********', self.__class__)
        logger.info(self,'* General Info')
        logger.info(self,'Date %s' % time.ctime())
        logger.info(self,'Python %s' % sys.version)
        logger.info(self,'Numpy %s' % numpy.__version__)
        logger.info(self,'Number of threads %d' % self.nthreads)
        logger.info(self,'Verbose level %d' % self.verbose)
        logger.info(self,'Scratch dir %s' % self.scratch)
        logger.info(self,'Input data file %s' % self.chkfile)
        logger.info(self,'Max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

        logger.info(self,'* Cell Info')
        logger.info(self,'Lattice vectors (Bohr)')
        for i in range(3):
            logger.info(self,'Cell a%d axis : %.6f  %.6f  %.6f', i, *self.a[i])
        logger.info(self,'Lattice reciprocal vectors (1/Bohr)')
        for i in range(3):
            logger.info(self,'Cell b%d axis : %.6f  %.6f  %.6f', i, *self.b[i])
        logger.info(self,'Cell volume %g (Bohr^3)', self.vol)
        logger.info(self,'Number of cell vectors %d' % len(self.ls))
        logger.info(self,'Number of kpoints %d ' % self.nkpts)
        for i in range(self.nkpts):
            logger.info(self,'K-point %d : %.6f  %.6f  %.6f', i, *self.kpts)
        logger.info(self,'Num atoms %d' % self.natm)
        logger.info(self,'Num electrons %d' % self.nelectron)
        logger.info(self,'Total charge %d' % self.charge)
        logger.info(self,'Spin %d ' % self.spin)
        logger.info(self,'Atom Coordinates (Bohr)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.coords[i])
        logger.info(self,'Atom Coordinates (Alat)')
        for i in range(self.natm):
            logger.info(self,'Nuclei %d with charge %d position : %.6f  %.6f  %.6f', i, 
                        self.charges[i], *self.frac[i])
        logger.info(self,'')

        return self

    def build(self):

        t0 = time.clock()
        lib.logger.TIMER_LEVEL = 3

        cell = libpbc.chkfile.load_cell(self.chkfile)
        self.cell = cell
        self.a = self.cell.lattice_vectors()
        self.b = self.cell.reciprocal_vectors()
        self.vol = self.cell.vol
        self.nelectron = self.cell.nelectron 
        self.charge = self.cell.charge    
        self.spin = self.cell.spin      
        self.natm = self.cell.natm		
        self.kpts = lib.chkfile.load(self.chkfile, 'kcell/kpts')
        self.nkpts = len(self.kpts)
        self.ls = cell.get_lattice_Ls(dimension=3)
        self.ls = self.ls[numpy.argsort(lib.norm(self.ls, axis=1))]
        self.coords = numpy.asarray([(numpy.asarray(atom[1])).tolist() for atom in cell._atom])
        self.charges = cell.atom_charges()
        self.frac = cart2frac(self.a,self.coords)            

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose > logger.NOTE:
            self.dump_input()

        spginfo(self)

        logger.timer(self,'Spg build', t0)

        return self

    kernel = build

if __name__ == '__main__':
    name = 'gamma.chk'
    surf = Spg(name)
    surf.verbose = 4
    surf.kernel()

