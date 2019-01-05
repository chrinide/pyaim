module mod_gaunt

  use iso_c_binding, only: c_int, c_double, c_size_t
  implicit none
  private

  integer, parameter :: ip = c_int
  integer, parameter :: size_t = c_size_t
  integer, parameter :: rp = c_double

  real(kind=rp), dimension(:,:), allocatable :: app
  real(kind=rp), dimension(:,:), allocatable :: bpp

  public :: eval_gaunt

contains

  subroutine eval_gaunt (lmax) bind(c)

    implicit none
    integer(kind=ip), intent(in), value :: lmax

    call allocate_space_for_gaunt (lmax)
    call init_gaunt (lmax)
    call deallocate_space_for_gaunt ()

  end subroutine eval_gaunt

  subroutine init_gaunt (lmax)

    implicit none
    integer(kind=ip), intent(in) :: lmax

    real(kind=rp) :: aux, bux, t1, t2
    integer(kind=ip) :: k1, k2, k20, k200, kk, kk0, kk00
    integer(kind=ip) :: mmp, n, mp, l, lp, m, m1 

    ! Computation of the app matrix 
    ! starting elements app(lm,00)(n) = delta(l,n)
    k1 = 0
    do l = 0,lmax
      do m = 0,l
        kk = k1*(k1+1)/2
        app(kk,l) = 1.0
        k1 = k1 + 1
      end do
    end do
    ! elements type app(lm,m'm')(n)
    do mp = 1,lmax
      k2 = mp*(mp+1)/2 + mp
      k20 = (mp-1)*mp/2 + mp-1
      do l = mp,lmax
        if (l.eq.mp) then
           m1 = mp
        else
           m1 = 0
        end if
        do m = m1,l
          k1 = l*(l+1)/2 + m
          kk = k1*(k1+1)/2 + k2
          kk0 = k1*(k1+1)/2 + k20
          do n = l-mp,l+mp,2
            if ( n.ge.m+mp) then
              app(kk,n) = (2*mp-1)*(app(kk0,n-1)/real(n+n-1,rp) - app(kk0,n+1)/real(n+n+3,rp))
            end if
          end do
        end do
      end do
    end do
    ! elements type app(lm,l'm')(n)
    do mp = 0,lmax
      do lp = mp+1,lmax
        k2 = lp*(lp+1)/2 + mp
        k20 = (lp-1)*lp/2 + mp
        k200 = (lp-2)*(lp-1)/2 + mp
        do l = lp,lmax
          if (l.eq.lp) then
            m1 = mp
          else
            m1 = 0
          end if
          do m = m1,l
            k1 = l*(l+1)/2 + m
            kk = k1*(k1+1)/2 + k2
            kk0 = k1*(k1+1)/2 + k20
            kk00 = k1*(k1+1)/2 + k200
            do n = l-lp,l+lp,2
              if (n.ge.m+mp) then
                aux = app(kk0,n+1)*(n+m+mp+1)/real(n+n+3,rp)
                if (n.gt.m+mp) aux = aux + app(kk0,n-1)*(n-m-mp)/real(n+n-1,rp)
                aux = aux*(lp+lp-1)
                if (lp.gt.mp+1) aux = aux - (lp+mp-1)*app(kk00,n)
                app(kk,n) = aux/real(lp-mp,rp)
              end if
            end do
          end do
        end do
      end do
    end do

    ! Computation bpp Matrix
    ! starting elements bpp(lm,00)(n) = delta(l,n)
    k1 = 0
    do l = 0,lmax
      do m = 0,l
        kk = k1*(k1+1) / 2
        bpp(kk,l) = 1.0
        k1 = k1 + 1
      end do
    end do
    ! elements type bpp(lm,m'm')(n)
    do mp = 1,lmax
      k2 = mp*(mp+1)/2 + mp
      k20 = (mp-1)*mp/2 + mp-1
      do l = mp,lmax
        if ( l.eq.mp ) then
          m1 = mp
        else
          m1 = 0
        end if
        do m = m1,l
          k1 = l*(l+1)/2 + m
          kk = k1*(k1+1)/2 + k2
          kk0 = k1*(k1+1)/2 + k20
          do n = l-mp,l+mp,2
            if (mp.gt.m) then
              t1 = 1.0
              t2 = 1.0
            else
              t1 = -(n-(m-mp+1))*(n-(m-mp+1)+1)
              t2 = -(n+(m-mp+1))*(n+(m-mp+1)+1)
            end if
            if (n.ge.abs(m-mp)) then
              if (n.eq.0) then
                bux = 0.0
              else
                bux = t1*bpp(kk0,n-1)/dfloat(n+n-1)
              end if
              bpp(kk,n) = (2*mp-1)*(bux-t2*bpp(kk0,n+1)/real(n+n+3,rp))
            end if
          end do
        end do
      end do
    end do
    ! elements type bpp(lm,l'm')(n)
    do mp = 0,lmax
      do lp = mp+1,lmax
        k2 = lp*(lp+1)/2 + mp
        k20 = (lp-1)*lp/2 + mp
        k200 = (lp-2)*(lp-1)/2 + mp
        do l = lp,lmax
          if (l.eq.lp) then
            m1 = mp
          else
            m1 = 0
          end if
          do m = m1,l
            k1 = l*(l+1)/2 + m
            kk = k1*(k1+1)/2 + k2
            kk0 = k1*(k1+1)/2 + k20
            kk00 = k1*(k1+1)/2 + k200
            do n = l-lp,l+lp,2
              mmp = abs(m-mp)
              if (n.ge.mmp) then
                aux = bpp(kk0,n+1)*(n+mmp+1)/real(n+n+3,rp)
                if (n.gt.mmp) aux = aux + bpp(kk0,n-1)*(n-mmp)/real(n+n-1,rp)
                aux = aux*(lp+lp-1)
                if (lp.gt.mp+1) aux = aux - (lp+mp-1)*bpp(kk00,n)
                bpp(kk,n) = aux/real(lp-mp,rp)
              end if
            end do
          end do
        end do
      end do
    end do

  end subroutine

  subroutine allocate_space_for_gaunt (maxl)
  integer(kind=ip), intent(in) :: maxl
  integer(kind=ip) :: mxlcof, mxkcof
  integer(kind=ip) :: ier
  mxlcof = maxl*(maxl+3)/2
  mxkcof = mxlcof*(mxlcof+3)/2 
  allocate (app(0:mxkcof,0:2*maxl+1) ,stat=ier) 
  if (ier.ne.0) stop "cannot alloc memory"
  allocate (bpp(0:mxkcof,0:2*maxl+1) ,stat=ier) 
  if (ier.ne.0) stop "cannot alloc memory"
  end subroutine allocate_space_for_gaunt
  
  subroutine deallocate_space_for_gaunt ()
  integer(kind=ip) :: ier
  deallocate (app,stat=ier) 
  deallocate (bpp,stat=ier) 
  end subroutine deallocate_space_for_gaunt

end module mod_gaunt 
