#!/usr/bin/env make

CC = gcc
FC = gfortran
LD = gcc
CDEBUG = -g
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm 
FFLAGS = -O3 -mtune=native -fopenmp -fpic 
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto -lgfortran

all: libaim.so

COBJECTS = surf.o 
FOBJECTS = mod_slm.o 

libaim.so: $(COBJECTS) $(FOBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(COBJECTS) $(FOBJECTS)

clean:
	/bin/rm -f *.so *.o *.pyc 

surf.o: surf.h

.SUFFIXES:
.SUFFIXES: .c .o .f90

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

.f90.o:
	$(FC) -c $(FFLAGS) -o $@ $<

