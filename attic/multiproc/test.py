#!/usr/bin/env python
from timeit import default_timer as timer
import multiprocessing as mp
#from multiprocessing.dummy import Pool as ThreadPool
##proc = mp.cpu_count()
##print proc
pool = mp.Pool(processes=2)
#pool = ThreadPool(8)
#a = [0.0, 0.1, 1.0]
#a = numpy.asarray(a)
#a = numpy.reshape(a, (-1,3))
##results = [pool.apply_async(eval_grad, args=(a,)) for x in range(npoints)]
##output = [p.get() for p in results]
#t0 = time.clock()
start = timer()
b = []
for i in range(npoints):
    b.append(a+i*0.001)
b = numpy.asarray(b)
##print b.shape
results = pool.map(do_some_job, b)
#results = pool.map_async(do_some_job, b)
#results.get()
end = timer()
print (end-start)
##print results
##output = [p.get() for p in results]
##print results[0]
#log.timer('pool', t0)

b = []
for i in range(npoints):
    b.append(a+i*0.001)
b = numpy.asarray(b)
start = timer()
val = map(do_some_job, b)
end = timer()
print (end-start)

#import pymp
#t0 = time.clock()
#start = timer ()
#with pymp.Parallel(1) as p:
#    for i in p.range(npoints):
#        #eval_grad(a)
#        do_some_job(a)
#end = timer()
#print (end-start)
#log.timer('own', t0)
