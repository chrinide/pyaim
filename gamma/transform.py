#!/usr/bin/python

import numpy

# Assumes the cell
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
# Assumes the cell
# | ax  ay  az |
# | bx  by  bz |
# | cx  cy  cz |
# use the transpose of this matrix
#
# Assumes coords are of the form:
# | x0  y0  z0 |
# | x1  y1  z1 |
# | x2  y2  z2 |
# | .......... |
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

natm = 36
cell = numpy.zeros((3,3))
cell[0] = [   9.16470408 ,  0.0000      , 0.000]
cell[1] = [ -4.51601841  , 7.91943334  , 0.000]
cell[2] = [ -0.00017949 , -0.00065987 , 29.54435511]
coords = numpy.zeros((36,3))
coords[0 ] = [-0.260289  ,  0.033695 ,   8.608132  ]
coords[1 ] = [-0.162668  , -0.096959 ,   3.619627  ]
coords[2 ] = [ 0.026984  ,  0.313231 ,  20.855301  ]
coords[3 ] = [-0.007111  , -0.326291 ,  25.995192  ]
coords[4 ] = [ 4.743181  ,  2.404431 ,  21.455790  ]
coords[5 ] = [ 4.331738  ,  2.387806 ,  11.597607  ]
coords[6 ] = [ 4.506493  ,  2.491378 ,  29.290894  ]
coords[7 ] = [ 4.378793  ,  2.414610 ,   4.662956  ]
coords[8 ] = [-0.260526  ,  4.984739 ,   0.632574  ]
coords[9 ] = [ 0.213877  ,  5.351617 ,  24.979309  ]
coords[10] = [  0.065941 ,   5.302108,    8.154685 ]
coords[11] = [-0.122101  ,  5.190246 ,  18.073405  ]
coords[12] = [  2.638904 ,  -0.189314,    6.044923 ]
coords[13] = [-1.658615  ,  2.500034 ,   6.246031  ]
coords[14] = [  3.062864 ,   5.534419,    6.294366 ]
coords[15] = [  6.466619 ,   0.069983,   23.452693 ]
coords[16] = [ -3.100452 ,   5.778139,   23.343290 ]
coords[17] = [  1.544057 ,   2.788424,   23.087791 ]
coords[18] = [  7.515691 ,   2.600887,   19.387301 ]
coords[19] = [  3.169490 ,   5.292150,   18.952113 ]
coords[20] = [  3.205328 ,   0.093494,   19.273792 ]
coords[21] = [  1.673518 ,   2.519977,    2.224158 ]
coords[22] = [  5.947234 ,   0.111577,    2.116646 ]
coords[23] = [  5.821929 ,   5.057659,    2.105035 ]
coords[24] = [  2.888806 ,   5.179482,   27.725605 ]
coords[25] = [ -1.404430 ,   7.553912,   27.632806 ]
coords[26] = [ -1.529703 ,   2.591210,   27.533212 ]
coords[27] = [ -3.038700 ,   5.305971,   10.726699 ]
coords[28] = [  1.394871 ,   2.728577,    9.952341 ]
coords[29] = [  1.393221 ,   7.671772,   10.505795 ]
coords[30] = [ -0.453593 ,   5.244937,   14.909588 ]
coords[31] = [ -0.990006 ,   9.199927,   15.579388 ]
coords[32] = [  3.886474 ,   2.805090,   14.758410 ]
coords[33] = [  6.864003 ,   5.531790,   12.658604 ]
coords[34] = [  3.601669 ,   4.600318,   17.095102 ]
coords[35] = [  0.690534 ,   4.106952,   13.964996 ]
print "#####"
frac = cart2frac(cell,coords)
print frac
