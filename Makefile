#!/usr/bin/env make

CC = gcc
LD = gcc
CDEBUG = -g
CFLAGS = -O3 -mtune=native -fopenmp -fpic -lm 
LFLAGS = -shared -L/home/jluis/src/pyscf/dev/pyscf/lib -lcgto

all: libaim.so

OBJECTS = surf.o 

libaim.so: $(OBJECTS)
	$(LD) $(LFLAGS) -o libaim.so $(OBJECTS)

clean:
	/bin/rm -f *.so *.o

surf.o: surf.h

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) -o $@ $<

