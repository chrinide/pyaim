module plot

  use iso_c_binding, only: c_int, c_double, c_char
  implicit none
  private

  integer, parameter, public :: ip = c_int
  integer, parameter, public :: rp = c_double

  real(kind=rp), parameter, public :: pi = 3.14159265358979323846264338328_rp
  real(kind=rp), parameter, public :: eps = 1d-7

contains

  !Di+ The following code writes atomic surface files in different
  !    formats for visualization purposes. It only works for
  !    regular THETA/PHI grids
  subroutine grasp_surface(x,y,z,mid,fname) bind(c)

    implicit none
    integer(kind=ip), intent(in), value :: npang
    integer(kind=c_char) :: fname
    real(kind=rp), intent(in), dimension(npang) :: x, y, z
    real(kind=rp), intent(in), dimension(3) :: mid
    call graspsurf(npang,npth,npph,x,y,z,mid,fname)

  end subroutine grasp_surface

  ! Di+ The following subroutines print out the GRASP atomic surface file 
  subroutine  graspsurf(npang,npth,npph,fneq,lsu,x,y,z,mid)
      integer(kind=ip) :: npang,npth,npph,lsu  
      real(kind=dp) :: x(npang), y(npang), z(npang) , mid(3)
      character(len=132) :: fneq
c     
      real(kind=dp),allocatable,dimension (:,:) :: vert,vnor
      integer,allocatable,dimension (:) :: vindx , nnorm 
      integer :: vtot,itot,itriang,ivert,nvert,ntriang,it,i,j,k,l
      integer :: inext1,inext2
      real(kind=dp) :: north(3),south(3),center(3),u(3)
      real(kind=dp) :: raver,rdist,unorm,rdummy
c
      character*80 line(5),fname,pname
c
c     vert ---> Cartesian coordinates of the vertex points
c     vnor  --->  average normal vector at the vertex point
c     vindx  ---> array of pointers to the vertex points constituting a triangle
c     
      ntriang=npph*(2*(npth-1))  +   2 * npph
      nvert=npang + 2
      vtot=nvert
      itot=ntriang
c     
      if (.not.allocated(vert)) then
         allocate (vert(3,vtot),stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot allocate array vert()'
      endif
      if (.not.allocated(vnor)) then
         allocate (vnor(3,vtot),stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot allocate array vnor()'
      endif
      if (.not.allocated(nnorm)) then
         allocate (nnorm(vtot),stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot allocate array vnor()'
      endif
      if (.not.allocated(vindx)) then
         allocate (vindx(3*itot),stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot allocate array vindx()'
      endif
c
      do i=1,vtot
      nnorm(i)=0
      do j=1,3
          vert(j,i)=0.0
          vnor(j,i)=0.0
      enddo
      enddo
c
      itriang=0
      ivert=0
c
      do i=1,npph 

        do it=1,npth-1
         
          ivert=ivert+1
          vert(1,ivert)=x(ivert)
          vert(2,ivert)=y(ivert)
          vert(3,ivert)=z(ivert)

          if ( i .lt. npph) then
             inext1=ivert+npth
             inext2=ivert+npth+1
          else
             inext1=it
             inext2=it+1
          endif

          itriang=itriang+1
          if ( itriang .gt. itot ) then
             print*,'itriang > itot' 
             stop
          endif
          vindx(3*(itriang-1) + 1 ) = ivert
          vindx(3*(itriang-1) + 2 ) = ivert+1
          vindx(3*(itriang-1) + 3 ) = inext2
          call surfnorm ( x , y, z, vnor, nnorm, nvert, 
     &                    ivert, ivert +1, inext2 )
          
          itriang=itriang+1
          if ( itriang .gt. itot ) then
             print*,'itriang > itot' 
             stop
          endif
          vindx(3*(itriang-1) + 1 ) = inext2
          vindx(3*(itriang-1) + 2 ) = inext1
          vindx(3*(itriang-1) + 3 ) = ivert
          call surfnorm ( x , y, z, vnor, nnorm, nvert, 
     &                    inext2 , inext1, ivert )

        enddo
        ivert=ivert+1
        vert(1,ivert)=x(ivert)
        vert(2,ivert)=y(ivert)
        vert(3,ivert)=z(ivert)
         
      enddo
c
c     North and south pole vertex points
c
c
c     The xyz coordinates of poles are estimated
c     by averaging the rim of closer PHI points 
c
c
      do j=1,3
         north(j)=0.0
         south(j)=0.0
      enddo
      do i=1,npph
         north(1)=north(1)+x(npth*(i-1) + 1 ) 
         north(2)=north(2)+y(npth*(i-1) + 1 ) 
         north(3)=north(3)+z(npth*(i-1) + 1 ) 
         south(1)=south(1)+x(npth*(i-1) + npth ) 
         south(2)=south(2)+y(npth*(i-1) + npth ) 
         south(3)=south(3)+z(npth*(i-1) + npth ) 
      enddo
      do j=1,3
         north(j)=north(j)/float(npph)
         south(j)=south(j)/float(npph) 
         center(j)=mid(j) 
      enddo
c
c     The position of the pole is (imperfectly) refined 
c     by adjusting its relative distance to the center
c     of coordinates
c
      raver=0.0
      do i=1,npph
         rdist=0.0
         do j=1,3
            rdist= rdist + ( vert(j,npth*(i-1)+1) - center(j) )**2
         enddo
         rdist=sqrt(rdist)
         raver=raver+rdist
      enddo
      raver=raver/float(npph)
      unorm=0.0
      do j=1,3
        u(j)=north(j)-center(j)
        unorm=unorm+u(j)**2
      enddo  
      unorm=sqrt(unorm)
      do j=1,3
        u(j)=u(j)/unorm
      enddo
      do j=1,3
        north(j)=center(j)+raver*u(j)
      enddo
c
c     The "refined" pole is added to the set of vertex points
c     and a new set of triangles is added
c
      ivert=ivert+1
      vert(1,ivert)=north(1)
      vert(2,ivert)=north(2)
      vert(3,ivert)=north(3)
      do i=1,npph 
c
        if ( i .lt. npph) then
             inext1=npth*(i-1) + 1 
             inext2=npth*i + 1
        else
             inext1=npth*(i-1) + 1 
             inext2=1
        endif
c
        itriang=itriang+1
        if ( itriang .gt. itot ) then
             print*,'itriang > itot' 
             stop
        endif
        vindx(3*(itriang-1) + 1 ) = ivert 
        vindx(3*(itriang-1) + 2 ) = inext1
        vindx(3*(itriang-1) + 3 ) = inext2
        call surfnorm ( x , y, z, vnor, nnorm, nvert, 
     &                ivert, inext1 , inext2 )
      enddo
c
c    Similar things with the south pole .....
c
      raver=0.0
      do i=1,npph
         rdist=0.0
         do j=1,3
            rdist= rdist + ( vert(j,npth*(i-1)+npth) - center(j) )**2
         enddo
         rdist=sqrt(rdist)
         raver=raver+rdist
      enddo
      raver=raver/float(npph)
      unorm=0.0
      do j=1,3
        u(j)=south(j)-center(j)
        unorm=unorm+u(j)**2
      enddo  
      unorm=sqrt(unorm)
      do j=1,3
        u(j)=u(j)/unorm
      enddo
      do j=1,3
        south(j)=center(j)+raver*u(j)
      enddo
      ivert=ivert+1
      vert(1,ivert)=south(1)
      vert(2,ivert)=south(2)
      vert(3,ivert)=south(3)
      do i=1,npph 
c
        if ( i .lt. npph) then
             inext1=npth*(i-1) + npth
             inext2=npth*i + npth
        else
             inext1=npth*(i-1) + npth 
             inext2=npth
        endif
c
        itriang=itriang+1
        if ( itriang .gt. itot ) then
             print*,'itriang > itot' 
             stop
        endif
        vindx(3*(itriang-1) + 1 ) = inext1
        vindx(3*(itriang-1) + 2 ) = ivert  
        vindx(3*(itriang-1) + 3 ) = inext2
        call surfnorm ( x , y, z, vnor, nnorm, nvert, 
     &                inext1, ivert, inext2 )
      enddo
c
      do i=1,nvert
         rdummy=0.0
         do j=1,3
            vnor(j,i)=vnor(j,i)/float(nnorm(i))
            rdummy=rdummy+vnor(j,i)**2
         enddo
         rdummy=sqrt(rdummy)
         do j=1,3
            vnor(j,i)=vnor(j,i)/rdummy 
         enddo
      enddo
c
c     Print out everything to the binary output file
c
      if ( itot .ne. itriang) then
         write(6,*) 'Problem in # of triangle faces'
         stop
      endif
      if ( vtot .ne. ivert) then
         write(6,*) 'Problem in # of vertex points'
         stop
      endif
c
      do i=1,5
          line(i)=" "
      end do
C
      line(1)="format=2"
      line(2)="vertices,normals,triangles"
      line(3)=" "
      igrid=65
      scale=0.33 
      write(line(4),'(3i6,f12.6)') vtot,itot,igrid,scale
      write(line(5),'(3f12.6)') sngl(mid)
c     
      open(lsu,file=fneq,form="unformatted")
c     
      do i=1,5
          write(lsu) line(i)
      enddo
c     
      write(lsu) sngl(vert)
      write(lsu) sngl(vnor)
      write(lsu) vindx
c     
      close(lsu)
c
      if (allocated(vert)) then
         deallocate (vert,stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot deallocate array vert()'
      endif
      if (allocated(vnor)) then
         deallocate (vnor,stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot deallocate array vnor()'
      endif
      if (allocated(vindx)) then
         deallocate (vindx,stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot deallocate array vindx()'
      endif
      if (allocated(nnorm)) then
         deallocate (nnorm,stat=ier)
         if (ier.ne.0) stop 'wrsurf.f: Cannot deallocate array nnorm()'
      endif
      return
      end
C
      subroutine surfnorm ( x , y, z, vnor, nnorm, nvert, i1, i2, i3) 
c
      USE       mod_prec, only: ip, dp 
c
      integer(kind=ip) ::  nvert, i1 i2, i3
      real(kind=dp) :: x(nvert), y(nvert), z(nvert) 

      real(kind=dp) :: vnor(3,nvert) 
      integer(kind=ip) ::  nnorm(nvert) 
c
      real(kind=dp) :: a(3),b(3),c(3) , cnorm
c
      a(1)=x(i2)-x(i1)
      a(2)=y(i2)-y(i1)
      a(3)=z(i2)-z(i1)
      b(1)=x(i3)-x(i1)
      b(2)=y(i3)-y(i1)
      b(3)=z(i3)-z(i1)
      call vecprod(a,b,c)
      cnorm=0.0
      do j=1,3
        cnorm=cnorm+c(j)**2
      enddo
      cnorm=sqrt(cnorm)
      do j=1,3
        c(j)=c(j)/cnorm 
      enddo
      nnorm(i1)=nnorm(i1)+1
      do j=1,3
         vnor(j,i1)=vnor(j,i1)+c(j)
      enddo
c
      b(1)=x(i3)-x(i2)
      b(2)=y(i3)-y(i2)
      b(3)=z(i3)-z(i2)
      call vecprod(a,b,c)
      cnorm=0.0
      do j=1,3
        cnorm=cnorm+c(j)**2
      enddo
      cnorm=sqrt(cnorm)
      do j=1,3
        c(j)=c(j)/cnorm 
      enddo
      nnorm(i2)=nnorm(i2)+1
      do j=1,3
         vnor(j,i2)=vnor(j,i2)+c(j)
      enddo
c
      a(1)=x(i1)-x(i3)
      a(2)=y(i1)-y(i3)
      a(3)=z(i1)-z(i3)
      call vecprod(b,a,c)
      cnorm=0.0
      do j=1,3
        cnorm=cnorm+c(j)**2
      enddo
      cnorm=sqrt(cnorm)
      do j=1,3
        c(j)=c(j)/cnorm 
      enddo
      nnorm(i3)=nnorm(i3)+1
      do j=1,3
         vnor(j,i3)=vnor(j,i3)+c(j)
      enddo
c
      return
      end

