#!/usr/bin/env python
#import matplotlib.pyplot as plt
import numpy
from scipy import stats
numpy.random.seed(12345678)
x = numpy.random.random(10)
y = numpy.random.random(10)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#To get coefficient of determination (r_squared)
print("r-squared:", r_value**2)
#plt.plot(x, y, 'o', label='original data')
#plt.plot(x, intercept + slope*x, 'r', label='fitted line')
#plt.legend()
#plt.show()
