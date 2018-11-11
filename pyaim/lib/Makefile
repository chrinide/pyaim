#!/usr/bin/env make

CC = gcc
LD = gcc
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm 
CDEBUG = -g -fpic -fopenmp
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto

all: libaim.so

OBJECTS = surf.o 

libaim.so: $(OBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(OBJECTS)

clean:
	/bin/rm *.so *.o

surf.o: surf.h

.SUFFIXES:
.SUFFIXES: .c .o .f90

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

