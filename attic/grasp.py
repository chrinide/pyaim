#!/usr/bin/env python
            do j=1,npang
               r=rlimsurf(i,j,nlimsurf(i,j))
               x(j)=r*st(j)*cp(j) + xyzrho(i,1)
               y(j)=r*st(j)*sp(j) + xyzrho(i,2)
               z(j)=r*ct(j) + xyzrho(i,3)
               x(j)=x(j)*0.5291772
               y(j)=y(j)*0.5291772
               z(j)=z(j)*0.5291772
            enddo
            do j=1,3
             mid(j)=xyzrho(i,j)
            enddo
