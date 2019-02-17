#!/usr/bin/env make

CC = gcc
LD = gcc
CDEBUG = -Wpedantic -g -pg -Wunused -Wall
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm -ftree-loop-vectorize -ffast-math #$(CDEBUG) 
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto -lpbc -lgfortran

all: libaim.so

COBJECTS = surf.o

libaim.so: $(COBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(COBJECTS)

clean:
	/bin/rm -f *.so *.o *.pyc *.mod

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

surf.o: surf.h

