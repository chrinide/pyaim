
  !> Integration using adaptive_stepper step.
  function adaptive_stepper(fid,xpoint,h0,maxstep,eps,res)
    use global, only: nav_stepper, nav_stepper_heun, nav_stepper_rkck, nav_stepper_dp,&
       nav_stepper_bs, nav_stepper_euler, nav_maxerr
    use types, only: scalar_value
    use param, only: vsmall
    logical :: adaptive_stepper
    type(field), intent(inout) :: fid
    real*8, intent(inout) :: xpoint(3)
    real*8, intent(inout) :: h0
    real*8, intent(in) :: maxstep, eps
    type(scalar_value), intent(inout) :: res

    integer :: ier
    real*8 :: grdt(3), ogrdt(3)
    real*8 :: xtemp(3), escalar, xerrv(3)
    real*8 :: nerr
    logical :: ok, first

    real*8, parameter :: h0break = 1.d-10

    adaptive_stepper = .true.
    ier = 1
    grdt = res%gf / (res%gfmod + VSMALL)
    first = .true.
    do while (ier /= 0)
       ! calculate the new grdt
       ogrdt = grdt

       ! new point
       if (NAV_stepper == NAV_stepper_euler) then
          call stepper_euler1(xpoint,grdt,h0,xtemp)
       else if (NAV_stepper == NAV_stepper_heun) then
          call stepper_heun(fid,xpoint,grdt,h0,xtemp,res)
       else if (NAV_stepper == NAV_stepper_rkck) then
          call stepper_rkck(fid,xpoint,grdt,h0,xtemp,xerrv,res)
       else if (NAV_stepper == NAV_stepper_dp) then
          call stepper_dp(fid,xpoint,grdt,h0,xtemp,xerrv,res)
       else if (NAV_stepper == NAV_stepper_bs) then
          call stepper_bs(fid,xpoint,grdt,h0,xtemp,xerrv,res)
       end if

       ! FSAL for BS stepper
       if (NAV_stepper /= NAV_stepper_bs) then
          call fid%grd(xtemp,2,res)
       end if
       grdt = res%gf / (res%gfmod + VSMALL)

       ! poor man's adaptive step size in Euler
       if (NAV_stepper == NAV_stepper_euler .or. NAV_stepper == NAV_stepper_heun) then
          ! angle with next step
          escalar = dot_product(ogrdt,grdt)

          ! gradient eps in cartesian
          ok = (res%gfmod < 0.99d0*eps)

          ! Check if they differ in > 90 deg.
          if (escalar < 0.d0.and..not.ok) then
             if (abs(h0) >= h0break) then
                h0 = 0.5d0 * h0
                ier = 1
             else
                adaptive_stepper = .false.
                return
             end if
          else
             ! Accept point. If angle is favorable, take longer steps
             if (escalar > 0.9 .and. first) &
                h0 = dsign(min(abs(maxstep), abs(1.6d0*h0)),maxstep)
             ier = 0
             xpoint = xtemp
          end if
       else
          ! use the error estimate
          nerr = norm2(xerrv)
          if (nerr < NAV_maxerr) then
             ! accept point
             ier = 0
             xpoint = xtemp
             ! if this is the first time through, and the norm is very small, propose a longer step
             if (first .and. nerr < NAV_maxerr/10d0) &
                h0 = dsign(min(abs(maxstep), abs(1.6d0*h0)),maxstep)
          else
             ! propose a new shorter step using the error estimate
             h0 = 0.9d0 * h0 * NAV_maxerr / nerr
             if (abs(h0) < VSMALL) then
                adaptive_stepper = .false.
                return
             end if
          endif
       end if
       first = .false.
    enddo

  end function adaptive_stepper

  !> Euler stepper.
  subroutine stepper_euler1(xpoint,grdt,h0,xout)
    
    real*8, intent(in) :: xpoint(3), h0, grdt(3)
    real*8, intent(out) :: xout(3)
  
    xout = xpoint + h0 * grdt
  
  end subroutine stepper_euler1

  !> Heun stepper.
  subroutine stepper_heun(fid,xpoint,grdt,h0,xout,res)
    use types, only: scalar_value
    use param, only: vsmall
    
    type(field), intent(inout) :: fid
    real*8, intent(in) :: xpoint(3), h0, grdt(3)
    real*8, intent(out) :: xout(3)
    type(scalar_value), intent(inout) :: res
    
    real*8 :: ak2(3)

    xout = xpoint + h0 * grdt

    call fid%grd(xout,2,res)
    ak2 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + 0.5d0 * h0 * (ak2 + grdt)
  
  end subroutine stepper_heun

  !> Bogacki-Shampine embedded 2(3) method, fsal
  subroutine stepper_bs(fid,xpoint,grdt,h0,xout,xerr,res)
    use types, only: scalar_value
    use param, only: vsmall
    
    type(field), intent(inout) :: fid
    real*8, intent(in) :: xpoint(3), h0, grdt(3)
    real*8, intent(out) :: xout(3), xerr(3)
    type(scalar_value), intent(inout) :: res

    real*8, dimension(3) :: ak1, ak2, ak3, ak4

    ak1 = grdt

    xout = xpoint + h0 * (0.5d0*ak1)
    call fid%grd(xout,2,res)
    ak2 = res%gf / (res%gfmod+VSMALL)

    xout = xpoint + h0 * (0.75d0*ak2)
    call fid%grd(xout,2,res)
    ak3 = res%gf / (res%gfmod+VSMALL)

    xout = xpoint + h0 * (2d0/9d0*ak1 + 1d0/3d0*ak2 + 4d0/9d0*ak3)
    call fid%grd(xout,2,res)
    ak4 = res%gf / (res%gfmod+VSMALL)

    xerr = xpoint + h0 * (7d0/24d0*ak1 + 1d0/4d0*ak2 + 1d0/3d0*ak3 + 1d0/8d0*ak4) - xout

  end subroutine stepper_bs

  !> Runge-Kutta-Cash-Karp embedded 4(5)-order, local extrapolation.
  subroutine stepper_rkck(fid,xpoint,grdt,h0,xout,xerr,res)
    use types, only: scalar_value
    use param, only: vsmall
    
    type(field), intent(inout) :: fid
    real*8, intent(in) :: xpoint(3), grdt(3), h0
    real*8, intent(out) :: xout(3), xerr(3)
    type(scalar_value), intent(inout) :: res

    real*8, parameter :: B21=.2d0, &
         B31=3.d0/40.d0, B32=9.d0/40.d0,&
         B41=.3d0, B42=-.9d0, B43=1.2d0,&
         B51=-11.d0/54.d0, B52=2.5d0, B53=-70.d0/27.d0, B54=35.d0/27.d0,&
         B61=1631.d0/55296.d0,B62=175.d0/512.d0, B63=575.d0/13824.d0, B64=44275.d0/110592.d0, B65=253.d0/4096.d0,&
         C1=37.d0/378.d0, C3=250.d0/621.d0, C4=125.d0/594.d0, C6=512.d0/1771.d0,&
         DC1=C1-2825.d0/27648.d0, DC3=C3-18575.d0/48384.d0, DC4=C4-13525.d0/55296.d0, DC5=-277.d0/14336.d0, DC6=C6-.25d0
    real*8, dimension(3) :: ak2, ak3, ak4, ak5, ak6
    
    xout = xpoint + h0*B21*grdt

    call fid%grd(xout,2,res)
    ak2 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(B31*grdt+B32*ak2)

    call fid%grd(xout,2,res)
    ak3 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(B41*grdt+B42*ak2+B43*ak3)

    call fid%grd(xout,2,res)
    ak4 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(B51*grdt+B52*ak2+B53*ak3+B54*ak4)

    call fid%grd(xout,2,res)
    ak5 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(B61*grdt+B62*ak2+B63*ak3+B64*ak4+B65*ak5)

    call fid%grd(xout,2,res)
    ak6 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(C1*grdt+C3*ak3+C4*ak4+C6*ak6)
    xerr = h0*(DC1*grdt+DC3*ak3+DC4*ak4+DC5*ak5+DC6*ak6)

  end subroutine stepper_rkck

  !> Doermand-Prince embedded 4(5)-order, local extrapolation.
  subroutine stepper_dp(fid,xpoint,grdt,h0,xout,xerr,res)
    use types, only: scalar_value
    use param, only: vsmall
    
    type(field), intent(inout) :: fid
    real*8, intent(in) :: xpoint(3), grdt(3), h0
    real*8, intent(out) :: xout(3), xerr(3)
    type(scalar_value), intent(inout) :: res

    real*8, parameter :: dp_a(7,7) = reshape((/&
       0.0d0,  0d0,0d0,0d0,0d0,0d0,0d0,&
       1d0/5d0,         0.0d0,0d0,0d0,0d0,0d0,0d0,&
       3d0/40d0,        9d0/40d0,       0.0d0,0d0,0d0,0d0,0d0,&
       44d0/45d0,      -56d0/15d0,      32d0/9d0,        0d0,0d0,0d0,0d0,&
       19372d0/6561d0, -25360d0/2187d0, 64448d0/6561d0, -212d0/729d0,  0d0,0d0,0d0,&
       9017d0/3168d0,  -355d0/33d0,     46732d0/5247d0,  49d0/176d0,  -5103d0/18656d0, 0d0,0d0,&
       35d0/384d0,      0d0,            500d0/1113d0,    125d0/192d0, -2187d0/6784d0,  11d0/84d0,      0d0&
       /),shape(dp_a))
    real*8, parameter :: dp_b2(7) = (/5179d0/57600d0, 0d0, 7571d0/16695d0, 393d0/640d0,&
       -92097d0/339200d0, 187d0/2100d0, 1d0/40d0/)
    real*8, parameter :: dp_b(7) = (/ 35d0/384d0, 0d0, 500d0/1113d0, 125d0/192d0, &
       -2187d0/6784d0, 11d0/84d0, 0d0 /)
    real*8, parameter :: dp_c(7) = dp_b2 - dp_b
    real*8, dimension(3) :: ak2, ak3, ak4, ak5, ak6, ak7

    xout = xpoint + h0*dp_a(2,1)*grdt

    call fid%grd(xout,2,res)
    ak2 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(dp_a(3,1)*grdt+dp_a(3,2)*ak2)

    call fid%grd(xout,2,res)
    ak3 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(dp_a(4,1)*grdt+dp_a(4,2)*ak2+dp_a(4,3)*ak3)

    call fid%grd(xout,2,res)
    ak4 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(dp_a(5,1)*grdt+dp_a(5,2)*ak2+dp_a(5,3)*ak3+dp_a(5,4)*ak4)

    call fid%grd(xout,2,res)
    ak5 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(dp_a(6,1)*grdt+dp_a(6,2)*ak2+dp_a(6,3)*ak3+dp_a(6,4)*ak4+dp_a(6,5)*ak5)

    call fid%grd(xout,2,res)
    ak6 = res%gf / (res%gfmod+VSMALL)
    xout = xpoint + h0*(dp_b(1)*grdt+dp_b(2)*ak2+dp_b(3)*ak3+dp_b(4)*ak4+dp_b(5)*ak5+dp_b(6)*ak6)

    call fid%grd(xout,2,res)
    ak7 = res%gf / (res%gfmod+VSMALL)
    xerr = h0*(dp_c(1)*grdt+dp_c(2)*ak2+dp_c(3)*ak3+dp_c(4)*ak4+dp_c(5)*ak5+dp_c(6)*ak6+dp_c(7)*ak7)
    xout = xout + xerr

  end subroutine stepper_dp
