/*
 * Atoms in moleucles module
 * Author: Jose Luis Casals Sainz <jluiscasalssainz@gmail.com>
 */

#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
                   
#include "surf.h"

void surf_driver(int inuc, int npang, double *ct, double *st, 
                 double *cp, double *sp,  
                 int ntrial, double *rpru, double epsiscp,
                 double epsroot, double rmaxsurf, int backend,
                 double epsilon, double step, int mstep,
                 int cart, double *coord, int *atm, int natm, 
                 int *bas, int nbas, double *env, int nprim,
                 int *ao_loc,
                 double *mo_coeff, double *mo_occ, 
                 double *nlimsurf, double *rsurf){

  int i, j;

  // Setup surface info
  inuc_ = inuc;
  natm_ = natm;
  coord_ = coord;
  epsiscp_ = epsiscp;
  ntrial_ = ntrial;      
  npang_ = npang;
  epsroot_ = epsroot;
  rmaxsurf_ = rmaxsurf;
  backend_ = backend;
  epsilon_ = epsilon;
  step_ = step;
  mstep_ = mstep;
  rpru_ = rpru;
	rsurf_ = (double *) malloc(sizeof(double)*npang_*ntrial_);
  assert(rsurf_ != NULL);
	nlimsurf_ = (double *) malloc(sizeof(double)*npang_);
  assert(nlimsurf_ != NULL);
	ct_ = (double *) malloc(sizeof(double)*npang_);
  assert(ct_ != NULL);
  st_ = (double *) malloc(sizeof(double)*npang_);
  assert(st_ != NULL);
	cp_ = (double *) malloc(sizeof(double)*npang_);
  assert(cp_ != NULL);
	sp_ = (double *) malloc(sizeof(double)*npang_);
  assert(sp_ != NULL);
  for (i=0; i<npang_; i++){
    ct_[i] = ct[i];
    st_[i] = st[i];
    cp_[i] = cp[i];
    sp_[i] = sp[i];
  }
	xnuc_ = (double *) malloc(sizeof(double)*3);
  assert(xnuc_ != NULL);
  xnuc_[0] = coord_[inuc_*3+0];
  xnuc_[1] = coord_[inuc_*3+1]; 
  xnuc_[2] = coord_[inuc_*3+2];

  // Basis info
  nprim_ = nprim;
  nbas_= nbas;
  cart_ = cart;
  atm_ = atm;
  bas_ = bas;
  env_ = env;
	ao_ = (double *) malloc(sizeof(double)*nprim_*4);
  ao_loc_ = ao_loc;
	non0tab_ = (int8_t *) malloc(sizeof(int8_t)*nbas_);
  assert(xnuc_ != NULL);
  for (i=0; i<nbas_; i++){
    non0tab_[i] = 1.0;
  }
	shls_ = (int *) malloc(sizeof(int)*2);
  assert(shls_ != NULL);
  shls_[0] = 0;
  shls_[1] = nbas_;
  nmo_ = 0;
  for (i=0; i<nprim_; i++){
    if (mo_occ[i] != 0) nmo_ += 1;
  }
	mo_coeff_ = (double *) malloc(sizeof(double)*nmo_*nprim_);
  assert(mo_coeff_ != NULL);
	mo_occ_ = (double *) malloc(sizeof(double)*nmo_);
  assert(mo_occ_ != NULL);
	c0_ = (double *) malloc(sizeof(double)*nmo_);
	c1_ = (double *) malloc(sizeof(double)*nmo_);
	c2_ = (double *) malloc(sizeof(double)*nmo_);
	c3_ = (double *) malloc(sizeof(double)*nmo_);
  assert(c0_ != NULL);
  assert(c1_ != NULL);
  assert(c2_ != NULL);
  assert(c3_ != NULL);
  int k = 0;
	for (i=0; i<nprim_; i++){ // Orbital
    if (mo_occ[i] != 0){
      mo_occ_[k] = mo_occ[i];
		  for (j=0; j<nprim_; j++){
        mo_coeff_[k*nprim_+j] = mo_coeff[j*nprim_+i];
      }
      k += 1;
    }
	}

  print_info();
	for (i=0; i<npang_; i++){
    nlimsurf[i] = nlimsurf_[i];
	  for (j=0; j<ntrial_; j++){
      rsurf[i*ntrial+j] = rsurf_[i*ntrial_+j];
    }
  }

  free(mo_coeff_);
  free(mo_occ_);
  free(coord_);
  free(atm_);
  free(bas_);
  free(env_);
  free(rpru_);
  free(cp_);
  free(sp_);
  free(ct_);
  free(st_);
  free(xnuc_);
  free(non0tab_);
  free(shls_);
  free(ao_loc_);
  free(ao_);
  free(c0_);
  free(c1_);
  free(c2_);
  free(c3_);

}

inline void rhograd(double *point, double *rho, double *grad, double *gradmod){

  if (cart_ == 1) {
    aim_GTOval_cart_deriv1(1, shls_, ao_loc_, ao_, point, 
                       non0tab_, atm_, natm_, bas_, nbas_, env_);
  }
  else {
    aim_GTOval_sph_deriv1(1, shls_, ao_loc_, ao_, point, 
                      non0tab_, atm_, natm_, bas_, nbas_, env_);
  }

  int i, j;

  for (i=0; i<nmo_; i++){
    c0_[i] = 0.0;
    c1_[i] = 0.0;
    c2_[i] = 0.0;
    c3_[i] = 0.0;
    for (j=0; j<nprim_; j++){
      c0_[i] += ao_[j+nprim_*0]*mo_coeff_[i*nprim_+j];
      c1_[i] += ao_[j+nprim_*1]*mo_coeff_[i*nprim_+j];
      c2_[i] += ao_[j+nprim_*2]*mo_coeff_[i*nprim_+j];
      c3_[i] += ao_[j+nprim_*3]*mo_coeff_[i*nprim_+j];
    }
  }

  *rho = 0.0;
  grad[0] = 0.0;
  grad[1] = 0.0;
  grad[2] = 0.0;
  *gradmod = 0.0;

  for (i=0; i<nmo_; i++){
    *rho += c0_[i]*c0_[i]*mo_occ_[i];
    grad[0] += c1_[i]*c0_[i]*mo_occ_[i]*2.0;
    grad[1] += c2_[i]*c0_[i]*mo_occ_[i]*2.0;
    grad[2] += c3_[i]*c0_[i]*mo_occ_[i]*2.0;
  }

  *gradmod = grad[0]*grad[0];
  *gradmod += grad[1]*grad[1];
  *gradmod += grad[2]*grad[2];
  *gradmod = sqrt(*gradmod);
  grad[0] = grad[0]/(*gradmod + HMINIMAL);
  grad[1] = grad[1]/(*gradmod + HMINIMAL);
  grad[2] = grad[2]/(*gradmod + HMINIMAL);

}

bool checkcp(double *x, double rho, double gradmod, int *nuc){

  int i;
  bool iscp = false;
  *nuc = -2;

    //printf("%.6f %.6f %.6f\n",coord_[0],coord_[1],coord_[2]);
    //printf("%.6f %.6f %.6f\n",coord_[3],coord_[4],coord_[5]);
  for (i=0; i<natm_; i++){
    double r = 0.0;
    r =  (x[0]-coord_[i*3+0])*(x[0]-coord_[i*3+0]);
    r += (x[1]-coord_[i*3+1])*(x[1]-coord_[i*3+1]);
    r += (x[2]-coord_[i*3+2])*(x[2]-coord_[i*3+2]);
    r = sqrt(r);
    //printf("%.6f %.6f %.6f %.6f\n", x[0], x[1], x[2], r);
    if (r <= epsiscp_){
      iscp = true;
      *nuc = i;
      //printf("%d %.6f %.6f\n", *nuc, rho, gradmod);
      return iscp;
    }
  }

  if (gradmod <= GRADEPS){
    iscp = true;
    if (rho <= RHOEPS){
      *nuc = -1;
    }
  } 

  return iscp;
  
}

//Runge-Kutta-Cash-Karp
inline void stepper_rkck(double *xpoint, double *grdt, double h0, double *xout, double *xerr){

  static const double b21 = 1.0/5.0;
  static const double b31 = 3.0/40.0;
  static const double b32 = 9.0/40.0;
  static const double b41 = 3.0/10.0;
  static const double b42 = -9.0/10.0;
  static const double b43 = 6.0/5.0;
  static const double b51 = -11.0/54.0;
  static const double b52 = 5.0/2.0;
  static const double b53 = -70.0/27.0;
  static const double b54 = 35.0/27.0;
  static const double b61 = 1631.0/55296.0;
  static const double b62 = 175.0/512.0;
  static const double b63 = 575.0/13824.0;
  static const double b64 = 44275.0/110592.0;
  static const double b65 = 253.0/4096.0;
  static const double c1 = 37.0/378.0;
  static const double c3 = 250.0/621.0;
  static const double c4 = 125.0/594.0;
  static const double c6 = 512.0/1771.0;
  static const double dc1 = c1-(2825.0/27648.0);
  static const double dc3 = c3-(18575.0/48384.0);
  static const double dc4 = c4-(13525.0/55296.0);
  static const double dc5 = -277.0/14336.0;
  static const double dc6 = c6-(1.0/4.0);
  
  double rho, grad[3], gradmod;

  //printf("1-%.6f %.6f %.6f %.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2], 
  //grdt[0], grdt[1], grdt[2]);

  xout[0] = xpoint[0] + h0*b21*grdt[0];
  xout[1] = xpoint[1] + h0*b21*grdt[1];
  xout[2] = xpoint[2] + h0*b21*grdt[2];
  //printf("2-%.6f %.6f %.6f %.6f %.6f\n", xout[0], xout[1], xout[2], h0, b21);

  double ak2[3] = {0.0};
  rhograd(xout, &rho, grad, &gradmod);
  ak2[0] = grad[0];
  ak2[1] = grad[1];
  ak2[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b31*grdt[0]+b32*ak2[0]);
  xout[1] = xpoint[1] + h0*(b31*grdt[1]+b32*ak2[1]);
  xout[2] = xpoint[2] + h0*(b31*grdt[2]+b32*ak2[2]);
  //printf("3-%.6f %.6f %.6f\n", xout[0], xout[1], xout[2]);
  
  double ak3[3] = {0.0};
  rhograd(xout, &rho, grad, &gradmod);
  ak3[0] = grad[0]; 
  ak3[1] = grad[1]; 
  ak3[2] = grad[2]; 
  xout[0] = xpoint[0] + h0*(b41*grdt[0]+b42*ak2[0]+b43*ak3[0]);
  xout[1] = xpoint[1] + h0*(b41*grdt[1]+b42*ak2[1]+b43*ak3[1]);
  xout[2] = xpoint[2] + h0*(b41*grdt[2]+b42*ak2[2]+b43*ak3[2]);
  //printf("4-%.6f %.6f %.6f\n", xout[0], xout[1], xout[2]);
  
  double ak4[3] = {0.0};
  rhograd(xout, &rho, grad, &gradmod);
  ak4[0] = grad[0];
  ak4[1] = grad[1];
  ak4[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b51*grdt[0]+b52*ak2[0]+b53*ak3[0]+b54*ak4[0]);
  xout[1] = xpoint[1] + h0*(b51*grdt[1]+b52*ak2[1]+b53*ak3[1]+b54*ak4[1]);
  xout[2] = xpoint[2] + h0*(b51*grdt[2]+b52*ak2[2]+b53*ak3[2]+b54*ak4[2]);
  
  double ak5[3] = {0.0};
  rhograd(xout, &rho, grad, &gradmod);
  ak5[0] = grad[0];
  ak5[1] = grad[1];
  ak5[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b61*grdt[0]+b62*ak2[0]+b63*ak3[0]+b64*ak4[0]+b65*ak5[0]);
  xout[1] = xpoint[1] + h0*(b61*grdt[1]+b62*ak2[1]+b63*ak3[1]+b64*ak4[1]+b65*ak5[1]);
  xout[2] = xpoint[2] + h0*(b61*grdt[2]+b62*ak2[2]+b63*ak3[2]+b64*ak4[2]+b65*ak5[2]);
  
  double ak6[3] = {0.0};
  rhograd(xout, &rho, grad, &gradmod);
  ak6[0] = grad[0];
  ak6[1] = grad[1];
  ak6[2] = grad[2];
  xout[0] = xpoint[0] + h0*(c1*grdt[0]+c3*ak3[0]+c4*ak4[0]+c6*ak6[0]);
  xout[1] = xpoint[1] + h0*(c1*grdt[1]+c3*ak3[1]+c4*ak4[1]+c6*ak6[1]);
  xout[2] = xpoint[2] + h0*(c1*grdt[2]+c3*ak3[2]+c4*ak4[2]+c6*ak6[2]);

  xerr[0] = h0*(dc1*grdt[0]+dc3*ak3[0]+dc4*ak4[0]+dc5*ak5[0]+dc6*ak6[0]);
  xerr[1] = h0*(dc1*grdt[1]+dc3*ak3[1]+dc4*ak4[1]+dc5*ak5[1]+dc6*ak6[1]);
  xerr[2] = h0*(dc1*grdt[2]+dc3*ak3[2]+dc4*ak4[2]+dc5*ak5[2]+dc6*ak6[2]);

}

bool adaptive_stepper(double *x, double *grad, double *h){

  int ier = 1;
  bool adaptive = true;
  double xout[3], xerr[3];

  while (ier != 0){
    double nerr = 0.0;
    stepper_rkck(x, grad, *h, xout, xerr);
    nerr += xerr[0]*xerr[0];
    nerr += xerr[1]*xerr[1];
    nerr += xerr[2]*xerr[2];
    nerr = sqrt(nerr)/3.0;
    if (nerr <= epsilon_){
      ier = 0;
      x[0] = xout[0];
      x[1] = xout[1];
      x[2] = xout[2];
      if (nerr < epsilon_/10.0){ 
        *h = fmin(MAXSTEP, ENLARGE*(*h));
      }
    } 
    else {
      double scale = SAFETY*(epsilon_/nerr);
      *h = scale*(*h);
      if (fabs(*h) < MINSTEP){
        adaptive = false;
        return adaptive; 
      }
    }
  }

  return adaptive;

}

//ier = 0 (correct), 1 (short step), 2 (too many iterations), 
//      3 (infty), 4 (nna)
int odeint(double *xpoint, double *rho, double *gradmod){

  int ier = 0, i;
  double h0 = step_;
  double grad[3] = {0.0};

	rhograd(xpoint, rho, grad, gradmod);
  //printf("1-%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2],
  //rho, grad[0],grad[1],grad[2],gradmod);

  if (*rho <= GRADEPS && *gradmod <= RHOEPS){
    ier = 3;
    return ier;
  }

  for (i=0; i<mstep_; i++){
    double xlast[3];
    xlast[0] = xpoint[0];
    xlast[1] = xpoint[1];
    xlast[2] = xpoint[2];
    //printf("2-%.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2]);
    bool ok = adaptive_stepper(xpoint, grad, &h0);
    //printf("3-%d %.6f %.6f %.6f\n", ok, xpoint[0], xpoint[1], xpoint[2]);
    if (ok == false){
      xpoint[0] = xlast[0];
      xpoint[1] = xlast[1];
      xpoint[2] = xlast[2];
      ier = 1;
      //exit(0);
      //printf("OK not");
      return ier;
    }
    //double trho, tgradmod;
	  rhograd(xpoint, rho, grad, gradmod);
    //printf("4-%.6f %.6f %.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2]), &rho, &gradmod;
    int nuc;
    bool iscp = checkcp(xpoint, *rho, *gradmod, &nuc);
    //printf("5-%d %d %.6f %.6f %.6f %.6f %.6f\n", iscp, nuc, xpoint[0], xpoint[1], xpoint[2]), &rho, &gradmod;
    if (iscp == true){
      ier = 0;
      return ier;
    }
  }

  double dist = 0.0;
  dist = (xpoint[0]-xnuc_[0])*(xpoint[0]-xnuc_[0]);
  dist += (xpoint[1]-xnuc_[1])*(xpoint[1]-xnuc_[1]);
  dist += (xpoint[2]-xnuc_[2])*(xpoint[2]-xnuc_[2]);
  dist = sqrt(dist);
  if (dist < rmaxsurf_){
    ier = 4;
  } 
  else if (dist >= rmaxsurf_){
    ier = 3;
  }
  else{
    ier = 2;
  }
    
  return ier;

}

void print_info(){

  int i;

  //printf("Follow info for surface_driver\n");
  //printf("\n"); 
  //printf("General info\n");
  //printf("###############################################\n");
  //printf("Machine precision : %e\n", HMINIMAL);
  //printf("Number of atoms : %d\n", natm_);
  //printf("Coordinates of atoms : \n");
	//for (i=0; i<natm_; i++){
  //  printf("%8.5f %8.5f %8.5f\n", coord_[i*natm_], coord_[i*natm_+1], coord_[i*natm_+2]);
	//}
  //printf("\n"); 
  //printf("Surface info\n");
  //printf("###############################################\n");
  //printf("Finding surface for atom : %d\n", inuc_);
  //printf("Coordinates of atom : %8.5f %8.5f %8.5f\n", xnuc_[0], xnuc_[1], xnuc_[2]);
  //printf("Gradient tolerance : %e\n", GRADEPS);
  //printf("Angular points in the mesh : %d\n", npang_);
  //printf("Max limit for surface : %8.5f\n", rmaxsurf_);
  //printf("Number of initial steeps in r-mesh : %d\n", ntrial_);
  //printf("R-mesh values \n"); 
	//for (i=0; i<ntrial_; i++){
  //  printf("%8.5f ", rpru_[i]);
  //}
  //printf("\n"); 
  //printf("Save distance for CP : %8.5f \n", epsiscp_);
  //printf("Bisection preccision : %8.5f \n", epsroot_);
  //printf("ODE solver : %d\n", backend_);
  //printf("ODE preccison : %8.5f\n", epsilon_);
  //printf("ODE first step : %8.5f\n", step_);
  //printf("ODE maximum iters : %d\n", mstep_);
  //printf("\n"); 
  //printf("Basis info\n");
  //printf("###############################################\n");
  //printf("Cart or spherical coordiantes : %d\n", cart_);
  //printf("Number of basis functions : %d\n", nbas_);
  //printf("Location of each AO\n");
	//for (i=0; i<nbas_; i++){
  //  printf("%d ", ao_loc_[i]);
  //}
  //printf("\n"); 
  //printf("Number of primitive functions : %d\n", nprim_);
  //printf("Number of occupied orbitals : %d\n", nmo_);
  //printf("MO occupations : \n");
  //printf("\n"); 
  //
  //printf("RHO eval info\n");
  //printf("###############################################\n");
  //double point[3];
  //double rho, grad[3], gradmod;
  //point[0] = -0.66875345;
  //point[1] =  0.;        
  //point[2] = -0.06767419; 
	//rhograd(point, &rho, grad, &gradmod);
  //printf("The value of rho is     : %12.8f \n", rho);
  //printf("The value of gradx is   : %12.8f \n", grad[0]);
  //printf("The value of grady is   : %12.8f \n", grad[1]);
  //printf("The value of gradz is   : %12.8f \n", grad[2]);
  //printf("The value of modgrad is : %12.8f \n", gradmod);
  //point[0] = 0.0;
  //point[1] = 0.0; 
  //point[2] = -0.73884;
	//rhograd(point, &rho, grad, &gradmod);
  //int nuc;
  //bool iscp = checkcp(point,rho,gradmod,&nuc);
  //printf("The value of iscp is : %d \n", iscp);
  //printf("The value of nuc is  : %d \n", nuc);
  //point[0] = 0.0;
  //point[1] = 0.0; 
  //point[2] = 0.0; 
	//rhograd(point, &rho, grad, &gradmod);
  //double h0 = 0.1, xout[3], xerr[3];
  //stepper_rkck(point, grad, h0, xout, xerr);
  //printf("The value of h0 is   : %12.8f \n", h0);
  //printf("The value of xout is : %12.8f \n", xout[0]);
  //printf("The value of yout is : %12.8f \n", xout[1]);
  //printf("The value of zout is : %12.8f \n", xout[2]);
  //printf("The value of xerr is : %12.8f \n", xerr[0]);
  //printf("The value of yerr is : %12.8f \n", xerr[1]);
  //printf("The value of zerr is : %12.8f \n", xerr[2]);
  //bool adaptive = adaptive_stepper(point, grad, &h0);
  //printf("The value of adaptive is : %d \n", adaptive);
  //printf("The value of h is        : %12.8f \n", h0);
  //double *xpoint; 
	//xpoint = (double *) malloc(sizeof(double)*3);
  //xpoint[0] = -0.740658; 
  //xpoint[1] =  0.000000; 
  //xpoint[2] =  0.001822; 
  //int ier = odeint(xpoint, &rho, &gradmod);
  //printf("The value of ier is     : %d     \n", ier);
  //printf("The value of xout is    : %12.8f \n", xpoint[0]);
  //printf("The value of yout is    : %12.8f \n", xpoint[1]);
  //printf("The value of zout is    : %12.8f \n", xpoint[2]);
  //printf("The value of rho is     : %12.8f \n", rho);
  //printf("The value of gradmod is : %12.8f \n", gradmod);
  //ier = odeint(xpoint, &rho, &gradmod);
  //printf("The value of ier is     : %12.8f \n", ier);
  //printf("The value of xout is    : %12.8f \n", xpoint[0]);
  //printf("The value of yout is    : %12.8f \n", xpoint[1]);
  //printf("The value of zout is    : %12.8f \n", xpoint[2]);
  //printf("The value of rho is     : %12.8f \n", rho);
  //printf("The value of gradmod is : %12.8f \n", gradmod);
  surface();

}

/*
int *mat = (int *)malloc(rows * cols * sizeof(int));
Then, you simulate the matrix using
int offset = i * cols + j;
// now mat[offset] corresponds to m(i, j)
for row-major ordering and
int offset = i + rows * j;
// not mat[offset] corresponds to m(i, j)
*/

void surface(){

  int i,j;
  double xsurf[ntrial_][3]; 
  int isurf[ntrial_][2];
  int nintersec = 0;
  double xpoint[3];

  if (natm_ == 1){  
	  for (i=0; i<npang_; i++){
      nlimsurf_[i] = 1;
      int offset = i*ntrial_;
      rsurf_[offset] = rmaxsurf_;
      return;
    }
  }

  double cost, sintcosp, sintsinp, rho, gradmod, ra, ract, rb;
  int ia, ib;

	for (i=0; i<npang_; i++){
    //if (i==18) exit(0);
    nintersec = 0;
    cost = ct_[i];
    sintcosp = st_[i]*cp_[i];
    sintsinp = st_[i]*sp_[i];
    ia = inuc_;
    ra = 0.0;
	  for (j=0; j<ntrial_; j++){
      ract = rpru_[j];
      xpoint[0] = xnuc_[0] + ract*sintcosp; 
      xpoint[1] = xnuc_[1] + ract*sintsinp; 
      xpoint[2] = xnuc_[2] + ract*cost;     
      //TODO: Check for error
      //printf("Initial %.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2]);
      int ier = odeint(xpoint, &rho, &gradmod);
      bool good = checkcp(xpoint, rho, gradmod, &ib);
      //printf("Final %d %.6f %.6f %.6f %.6f %.6f\n", ib, rho, gradmod, xpoint[0], xpoint[1], xpoint[2]);
      rb = ract;
      if (ib != ia && (ia == inuc_ || ib == inuc_)){
        if (ia != inuc_ || ib != -1){
          nintersec += 1;
          xsurf[nintersec-1][0] = ra;
          xsurf[nintersec-1][1] = rb;
          isurf[nintersec-1][0] = ia;
          isurf[nintersec-1][1] = ib;
        }                             
      }
      ia = ib;
      ra = rb;
    }
	  for (j=0; j<nintersec; j++){
      ia = isurf[j][0];
      ib = isurf[j][1];
      ra = xsurf[j][0];
      rb = xsurf[j][1];
      double xin[3], xfin[3];
      //printf("%d %d %.6f %.6f\n", ia, ib, ra, rb);
      xin[0] = xnuc_[0] + ra*sintcosp;
      xin[1] = xnuc_[1] + ra*sintsinp;
      xin[2] = xnuc_[2] + ra*cost;
      xfin[0] = xnuc_[0] + rb*sintcosp;
      xfin[1] = xnuc_[1] + rb*sintsinp;
      xfin[2] = xnuc_[2] + rb*cost;
      while (fabs(ra-rb) > epsroot_){
        double xmed[3];
        xmed[0] = 0.5*(xfin[0] + xin[0]);   
        xmed[1] = 0.5*(xfin[1] + xin[1]);   
        xmed[2] = 0.5*(xfin[2] + xin[2]);   
        double rm = 0.5*(ra + rb);
        xpoint[0] = xmed[0];
        xpoint[1] = xmed[1];
        xpoint[2] = xmed[2];
        int im;
        int ier = odeint(xpoint, &rho, &gradmod);
        //printf("%.6f %.6f %.6f\n", xpoint[0], xpoint[1], xpoint[2]);
        bool good = checkcp(xpoint, rho, gradmod, &im);
        //printf("%d %.6f %.6f %.6f %.6f %.6f\n", im, rho, gradmod, xpoint[0], xpoint[1], xpoint[2]);
        //if (ib != -1 && (im != ia && im != ib)){
        //  printf("warning new intersections found\n");
        //}
        if (im == ia){
          xin[0] = xmed[0];
          xin[1] = xmed[1];
          xin[2] = xmed[2];
          ra = rm;
        }
        else if (im == ib){
          xfin[0] = xmed[0];
          xfin[1] = xmed[1];
          xfin[2] = xmed[2];
          rb = rm;
        }
        else{
          if (ia == inuc_){
            xfin[0] = xmed[0];
            xfin[1] = xmed[1];
            xfin[2] = xmed[2];
            rb = rm;
          }
          else{
            xin[0] = xmed[0];
            xin[1] = xmed[1];
            xin[2] = xmed[2];
            ra = rm;
          }
        }
      }
      //xpoint[0] = 0.5*(xfin[0] + xin[0]);   
      //xpoint[1] = 0.5*(xfin[1] + xin[1]);   
      //xpoint[2] = 0.5*(xfin[2] + xin[2]);   
      xsurf[j][2] = 0.5*(ra + rb);
    }
    // organize pairs
    nlimsurf_[i] = nintersec;
    //printf("%6.f\n", nlimsurf_[i]);
	  for (j=0; j<nintersec; j++){
      int offset = i*ntrial_+j;
      rsurf_[offset] = xsurf[j][2];
    }
    if (nintersec%2 == 0){
      //printf("%d %d intermod\n", i, nintersec%2);
      nintersec = 1;
      nlimsurf_[i] += nintersec;
      int offset = i*ntrial_+(nintersec-1);
      rsurf_[offset] = rmaxsurf_;
    }
    printf("#* %d %d %.6f %.6f %.6f %.6f ",i,nintersec,ct_[i],st_[i],cp_[i],sp_[i]);
	  for (j=0; j<nintersec; j++){
     printf(" %.6f ",rsurf_[i*ntrial_+j]);
     //printf(" %.6f ",xsurf[j][2]);
    }
    printf("\n");
  }

}

