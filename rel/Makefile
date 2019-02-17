#!/usr/bin/env make

CC = gcc
LD = gcc
CDEBUG = -Wpedantic -g -pg -Wunused -Wall
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm -ftree-loop-vectorize -ffast-math #$(CDEBUG) 
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto -lgfortran

all: libaim.so lib4caim.so

COBJECTS = surf.o

libaim.so: $(COBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(COBJECTS)

lib4caim.so: surf4c.o
	$(LD) $(LFLAGS) -o lib4caim.so surf4c.o

clean:
	/bin/rm -f *.so *.o *.pyc *.mod

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

surf.o: surf.h
surf4c.o: surf.h

