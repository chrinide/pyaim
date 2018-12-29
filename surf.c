#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "surf.h"

void surf_driver(const int inuc, 
								 const double *xyzrho,
								 const int npang, 
								 const double *ct, 
                 const double *st, 
                 const double *cp,
							   const double *sp,  
                 const int ntrial, 
                 const double *rpru, 
                 const double epsiscp,
                 const double epsroot, 
                 const double rmaxsurf, 
                 const int backend,
                 const double epsilon, 
                 const double step, 
                 const int mstep,
							   const int natm,
								 const double *coords,
                 const int cart,    
								 const int nmo,
								 const int nprims,
								 int *atm,
								 const int nbas,
                 int *bas,
								 double *env,
                 int *ao_loc,
                 const double *mo_coeff, 
							   const double *mo_occ, 
                 int *nlimsurf, double *rsurf){

  int i, j;

  // Setup surface info
  natm_ = natm;
	coords_ = (double *) malloc(sizeof(double)*natm_*3);
  assert(coords_ != NULL);
  for (i=0; i<natm_; i++) {
    coords_[i*3+0] = coords[i*3+0];
    coords_[i*3+1] = coords[i*3+1];
    coords_[i*3+2] = coords[i*3+2];
  }
  inuc_ = inuc;
  xnuc_[0] = xyzrho[0];
  xnuc_[1] = xyzrho[1]; 
  xnuc_[2] = xyzrho[2];
  epsiscp_ = epsiscp;
  ntrial_ = ntrial;      
  npang_ = npang;
  epsroot_ = epsroot;
  rmaxsurf_ = rmaxsurf;
  backend_ = backend;
  epsilon_ = epsilon;
  step_ = step;
  mstep_ = mstep;
	rpru_ = (double *) malloc(sizeof(double)*ntrial_);
  assert(rpru_ != NULL);
  for (i=0; i<ntrial_; i++){
    rpru_[i] = rpru[i];
  }
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
	rsurf_ = (double *) malloc(sizeof(double)*npang_*ntrial_);
  assert(rsurf_ != NULL);
	nlimsurf_ = (int *) malloc(sizeof(int)*npang_);
  assert(nlimsurf_ != NULL);

  // Basis info
  nprims_ = nprims;
  nbas_= nbas;
  cart_ = cart;
  atm_ = atm;
  bas_ = bas;
  env_ = env;
  ao_loc_ = ao_loc;
	non0tab_ = (int8_t *) malloc(sizeof(int8_t)*nbas_);
  assert(non0tab_ != NULL);
  for (i=0; i<nbas_; i++){
    non0tab_[i] = 1.0;
  }
	shls_ = (int *) malloc(sizeof(int)*2);
  assert(shls_ != NULL);
  shls_[0] = 0;
  shls_[1] = nbas_;
  nmo_ = 0;
  for (i=0; i<nmo; i++){
    if (mo_occ[i] != 0) nmo_ += 1;
  }
	mo_coeff_ = (double *) malloc(sizeof(double)*nmo_*nprims_);
  assert(mo_coeff_ != NULL);
	mo_occ_ = (double *) malloc(sizeof(double)*nmo_);
  assert(mo_occ_ != NULL);
  int k = 0;
	for (i=0; i<nmo; i++){ // Orbital
    if (mo_occ[i] != 0){
      mo_occ_[k] = mo_occ[i];
		  for (j=0; j<nprims_; j++){
        mo_coeff_[k*nprims_+j] = mo_coeff[j*nprims_+i];
      }
      k += 1;
    }
	}

  surface();
	for (i=0; i<npang_; i++){
    nlimsurf[i] = nlimsurf_[i];
	  for (j=0; j<ntrial_; j++){
      rsurf[i*ntrial_+j] = rsurf_[i*ntrial_+j];
    }
  }

  free(mo_coeff_);
  free(mo_occ_);
  free(coords_);
  free(rpru_);
  free(cp_);
  free(sp_);
  free(ct_);
  free(st_);
  free(non0tab_);
  free(shls_);
  //free(atm_);
  //free(bas_);
  //free(env_);
  //free(ao_loc_);

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

#pragma omp parallel default(none)  \
    private(i,nintersec,j,xpoint,xsurf,isurf) \
    shared(npang_,ct_,st_,cp_,sp_,xnuc_,inuc_,rpru_,\
    ntrial_,epsroot_,rsurf_,nlimsurf_,rmaxsurf_)
{
#pragma omp for schedule(dynamic) nowait
	for (i=0; i<npang_; i++){
    nintersec = 0;
    double cost = ct_[i];
    double sintcosp = st_[i]*cp_[i];
    double sintsinp = st_[i]*sp_[i];
    int ia = inuc_, ib;
    double ra = 0.0, rb;
    double rho, gradmod;
	  for (j=0; j<ntrial_; j++){
      double ract = rpru_[j];
      xpoint[0] = xnuc_[0] + ract*sintcosp; 
      xpoint[1] = xnuc_[1] + ract*sintsinp; 
      xpoint[2] = xnuc_[2] + ract*cost;     
      //TODO: Better Check for error
      int ier = odeint(xpoint, &rho, &gradmod);
      if (ier == 1) {
        printf("Too short step on Odeint\n");
        exit(-1);
      } else if (ier == 2) {
        printf("Too many steeps on Odeint\n");
        exit(-1);
			} else if (ier == 4) {
        printf("NNA on Odeint\n");
        //exit(-1);
			}
      bool good = checkcp(xpoint, rho, gradmod, &ib);
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
      	if (ier == 1) {
	        printf("Too short step on Odeint\n");
	        exit(-1);
	      } else if (ier == 2) {
	        printf("Too many steeps on Odeint\n");
	        exit(-1);
				} else if (ier == 4) {
	        printf("NNA on Odeint\n");
	        //exit(-1);
				}
        bool good = checkcp(xpoint, rho, gradmod, &im);
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
      xsurf[j][2] = 0.5*(ra + rb);
    }
    // organize pairs
    nlimsurf_[i] = nintersec; 
	  for (j=0; j<nintersec; j++){
      int offset = i*ntrial_+j;
      rsurf_[offset] = xsurf[j][2];
    }
    if (nintersec%2 == 0){
      nintersec += 1;
      nlimsurf_[i] = nintersec;
      int offset = i*ntrial_+(nintersec-1);
      rsurf_[offset] = rmaxsurf_;
    }
    //printf("#* %d %d %.6f %.6f %.6f %.6f ",i,nlimsurf_[i],ct_[i],st_[i],cp_[i],sp_[i]);
	  //for (j=0; j<nlimsurf_[i]; j++){
    // printf(" %.6f ",rsurf_[i*ntrial_+j]);
    //}
    //printf("\n");
  }
}

}

inline void rho_grad(double *point, double *rho, double *grad, double *gradmod){

	double ao_[nprims_*4];
  double c0_[nmo_],c1_[nmo_],c2_[nmo_],c3_[nmo_];

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
    for (j=0; j<nprims_; j++){
      c0_[i] += ao_[j+nprims_*0]*mo_coeff_[i*nprims_+j];
      c1_[i] += ao_[j+nprims_*1]*mo_coeff_[i*nprims_+j];
      c2_[i] += ao_[j+nprims_*2]*mo_coeff_[i*nprims_+j];
      c3_[i] += ao_[j+nprims_*3]*mo_coeff_[i*nprims_+j];
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

//ier = 0 (correct), 1 (short step), 2 (too many iterations), 
//      3 (infty), 4 (nna)
int odeint(double *xpoint, double *rho, double *gradmod){

  int ier = 0, i;
  double h0 = step_;
  double grad[3] = {0.0};

	rho_grad(xpoint, rho, grad, gradmod);

  if (*rho <= RHOEPS && *gradmod <= GRADEPS){
    ier = 3;
    return ier;
  }

  for (i=0; i<mstep_; i++){
    double xlast[3];
    xlast[0] = xpoint[0];
    xlast[1] = xpoint[1];
    xlast[2] = xpoint[2];
    bool ok = adaptive_stepper(xpoint, grad, &h0);
    if (ok == false){
      xpoint[0] = xlast[0];
      xpoint[1] = xlast[1];
      xpoint[2] = xlast[2];
      ier = 1;
      //exit(0);
      return ier;
    }
	  rho_grad(xpoint, rho, grad, gradmod);
    int nuc;
    bool iscp = checkcp(xpoint, *rho, *gradmod, &nuc);
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

bool checkcp(const double *x, const double rho, const double gradmod, int *nuc){

  int i;
  bool iscp = false;
  *nuc = -2;

  for (i=0; i<natm_; i++){
    double r = 0.0;
    r =  (x[0]-coords_[i*3+0])*(x[0]-coords_[i*3+0]);
    r += (x[1]-coords_[i*3+1])*(x[1]-coords_[i*3+1]);
    r += (x[2]-coords_[i*3+2])*(x[2]-coords_[i*3+2]);
    r = sqrt(r);
    if (r <= epsiscp_){
      iscp = true;
      *nuc = i;
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

bool adaptive_stepper(double *x, const double *grad, double *h){

  int ier = 1;
  bool adaptive = true;
  double xout[3], xerr[3];

  while (ier != 0){
    double nerr = 0.0;
    stepper_rkck(x, grad, *h, xout, xerr);
    //stepper_rkdp(x, grad, *h, xout, xerr);
    nerr += xerr[0]*xerr[0];
    nerr += xerr[1]*xerr[1];
    nerr += xerr[2]*xerr[2];
    nerr = sqrt(nerr)/3.0;
    //nerr = sqrt(nerr);
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

inline void stepper_rkdp(const double *xpoint, 
                         const double *grdt, 
                         const double h0, 
                         double *xout, double *xerr){

  static const double b21 = 1.0/5.0;
  static const double b31 = 3.0/40.0;
  static const double b32 = 9.0/40.0;
  static const double b41 = 44.0/45.0;
  static const double b42 = -56.0/15.0;
  static const double b43 = 32.0/9.0;
  static const double b51 = 19372.0/6561.0;
  static const double b52 = -25360.0/2187.0;
  static const double b53 = 64448.0/6561.0;
  static const double b54 = -212.0/729.0;
  static const double b61 = 9017.0/3168.0;
  static const double b62 = -355.0/33.0;
  static const double b63 = 46732.0/5247.0;
  static const double b64 = 49.0/176.0;
  static const double b65 = -5103.0/18656.0;
  static const double b71 = 35.0/384.0;
  static const double b73 = 500.0/1113.0;
  static const double b74 = 125.0/192.0;
  static const double b75 = -2187.0/6784.0;
  static const double b76 = 11.0/84.0;

  static const double c1 = 35.0/384.0;
  static const double c3 = 500.0/1113.0;
  static const double c4 = 125.0/192.0;
  static const double c5 = -2187.0/6784.0;
  static const double c6 = 11.0/84.0;
  static const double c7 = 0.0;

  static const double b1 = 5179.0/57600.0;
  static const double b3 = 7571.0/16695.0;
  static const double b4 = 393.0/640.0;
  static const double b5 = -92097.0/339200.0;
  static const double b6 = 187.0/2100.0;
  static const double b7 = 1.0/40.0;

  static const double dc1 = c1-b1;
  static const double dc3 = c3-b3;
  static const double dc4 = c4-b4;
  static const double dc5 = c5-b5;
  static const double dc6 = c6-b6;
  static const double dc7 = -b7;

  double rho, grad[3], gradmod;

  xout[0] = xpoint[0] + h0*b21*grdt[0];
  xout[1] = xpoint[1] + h0*b21*grdt[1];
  xout[2] = xpoint[2] + h0*b21*grdt[2];

  double ak2[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak2[0] = grad[0];
  ak2[1] = grad[1];
  ak2[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b31*grdt[0]+b32*ak2[0]);
  xout[1] = xpoint[1] + h0*(b31*grdt[1]+b32*ak2[1]);
  xout[2] = xpoint[2] + h0*(b31*grdt[2]+b32*ak2[2]);
  
  double ak3[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak3[0] = grad[0]; 
  ak3[1] = grad[1]; 
  ak3[2] = grad[2]; 
  xout[0] = xpoint[0] + h0*(b41*grdt[0]+b42*ak2[0]+b43*ak3[0]);
  xout[1] = xpoint[1] + h0*(b41*grdt[1]+b42*ak2[1]+b43*ak3[1]);
  xout[2] = xpoint[2] + h0*(b41*grdt[2]+b42*ak2[2]+b43*ak3[2]);

  double ak4[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak4[0] = grad[0];
  ak4[1] = grad[1];
  ak4[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b51*grdt[0]+b52*ak2[0]+b53*ak3[0]+b54*ak4[0]);
  xout[1] = xpoint[1] + h0*(b51*grdt[1]+b52*ak2[1]+b53*ak3[1]+b54*ak4[1]);
  xout[2] = xpoint[2] + h0*(b51*grdt[2]+b52*ak2[2]+b53*ak3[2]+b54*ak4[2]);

  double ak5[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak5[0] = grad[0];
  ak5[1] = grad[1];
  ak5[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b61*grdt[0]+b62*ak2[0]+b63*ak3[0]+b64*ak4[0]+b65*ak5[0]);
  xout[1] = xpoint[1] + h0*(b61*grdt[1]+b62*ak2[1]+b63*ak3[1]+b64*ak4[1]+b65*ak5[1]);
  xout[2] = xpoint[2] + h0*(b61*grdt[2]+b62*ak2[2]+b63*ak3[2]+b64*ak4[2]+b65*ak5[2]);

  double ak6[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak6[0] = grad[0];
  ak6[1] = grad[1];
  ak6[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b71*grdt[0]+b73*ak3[0]+b74*ak4[0]+b75*ak5[0]+b76*ak6[0]);
  xout[1] = xpoint[1] + h0*(b71*grdt[1]+b73*ak3[1]+b74*ak4[1]+b75*ak5[1]+b76*ak6[1]);
  xout[2] = xpoint[2] + h0*(b71*grdt[2]+b73*ak3[2]+b74*ak4[2]+b75*ak5[2]+b76*ak6[2]);

  double ak7[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak7[0] = grad[0];
  ak7[1] = grad[1];
  ak7[2] = grad[2];
  xerr[0] = h0*(dc1*grdt[0]+dc3*ak3[0]+dc4*ak4[0]+dc5*ak5[0]+dc6*ak6[0]+dc7*ak7[0]);
  xerr[1] = h0*(dc1*grdt[1]+dc3*ak3[1]+dc4*ak4[1]+dc5*ak5[1]+dc6*ak6[1]+dc7*ak7[1]);
  xerr[2] = h0*(dc1*grdt[2]+dc3*ak3[2]+dc4*ak4[2]+dc5*ak5[2]+dc6*ak6[2]+dc7*ak7[2]);
  xout[0] += xerr[0];
  xout[1] += xerr[1];
  xout[2] += xerr[2];

}                         

// Runge-Kutta-Cash-Karp
inline void stepper_rkck(const double *xpoint, 
                         const double *grdt, 
                         const double h0, 
                         double *xout, double *xerr){

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

  xout[0] = xpoint[0] + h0*b21*grdt[0];
  xout[1] = xpoint[1] + h0*b21*grdt[1];
  xout[2] = xpoint[2] + h0*b21*grdt[2];

  double ak2[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak2[0] = grad[0];
  ak2[1] = grad[1];
  ak2[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b31*grdt[0]+b32*ak2[0]);
  xout[1] = xpoint[1] + h0*(b31*grdt[1]+b32*ak2[1]);
  xout[2] = xpoint[2] + h0*(b31*grdt[2]+b32*ak2[2]);
  
  double ak3[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak3[0] = grad[0]; 
  ak3[1] = grad[1]; 
  ak3[2] = grad[2]; 
  xout[0] = xpoint[0] + h0*(b41*grdt[0]+b42*ak2[0]+b43*ak3[0]);
  xout[1] = xpoint[1] + h0*(b41*grdt[1]+b42*ak2[1]+b43*ak3[1]);
  xout[2] = xpoint[2] + h0*(b41*grdt[2]+b42*ak2[2]+b43*ak3[2]);
  
  double ak4[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak4[0] = grad[0];
  ak4[1] = grad[1];
  ak4[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b51*grdt[0]+b52*ak2[0]+b53*ak3[0]+b54*ak4[0]);
  xout[1] = xpoint[1] + h0*(b51*grdt[1]+b52*ak2[1]+b53*ak3[1]+b54*ak4[1]);
  xout[2] = xpoint[2] + h0*(b51*grdt[2]+b52*ak2[2]+b53*ak3[2]+b54*ak4[2]);
  
  double ak5[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
  ak5[0] = grad[0];
  ak5[1] = grad[1];
  ak5[2] = grad[2];
  xout[0] = xpoint[0] + h0*(b61*grdt[0]+b62*ak2[0]+b63*ak3[0]+b64*ak4[0]+b65*ak5[0]);
  xout[1] = xpoint[1] + h0*(b61*grdt[1]+b62*ak2[1]+b63*ak3[1]+b64*ak4[1]+b65*ak5[1]);
  xout[2] = xpoint[2] + h0*(b61*grdt[2]+b62*ak2[2]+b63*ak3[2]+b64*ak4[2]+b65*ak5[2]);
  
  double ak6[3] = {0.0};
  rho_grad(xout, &rho, grad, &gradmod);
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
