#!/usr/bin/env make

CC = gcc
FC = gfortran
LD = gcc
FDEBUG = -Wpedantic -g -pg -Wunused -fbacktrace -fcheck=bounds,mem,pointer,do,array-temps -Wall
CDEBUG = -Wpedantic -g -pg -Wunused -Wall
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm #$(CDEBUG) 
FFLAGS = -O3 -mtune=native -fopenmp -fpic     #$(FDEBUG) 
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto -lpbc -lgfortran

all: libaim.so

COBJECTS = surf.o

libaim.so: $(COBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(COBJECTS)

clean:
	/bin/rm -f *.so *.o *.pyc *.mod

.SUFFIXES:
.SUFFIXES: .c .o .f90

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f90.o:
	$(FC) -c $(FFLAGS) -o $@ $<

surf.o: surf.h

