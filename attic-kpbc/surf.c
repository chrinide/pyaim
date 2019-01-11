#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <stdlib.h>
#include <complex.h>
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
                 const double complex *mo_coeff, 
							   const double *mo_occ, 
		             const double occdrop,
                 const double *a,
                 const int nls,
                 const double *ls,
                 const int nkpts,
                 const double complex *explk,
                 const double *rcut,
						     const unsigned char *non0tab,
                 int *nlimsurf, double *rsurf){

  int i, j, k;
 
  // Setup surface info
  natm_ = natm;
	//printf("The number of atoms is %d\n", natm_);
	coords_ = (double *) malloc(sizeof(double)*natm_*3);
  assert(coords_ != NULL);
  for (i=0; i<natm_; i++) {
    coords_[i*3+0] = coords[i*3+0];
    coords_[i*3+1] = coords[i*3+1];
    coords_[i*3+2] = coords[i*3+2];
		//printf("Coordinate of atom %d %f %f %f\n", i, coords_[i*3+0], coords_[i*3+1], coords_[i*3+2]);
  }
  xyzrho_ = (double *) malloc(sizeof(double)*natm_*3);
  assert(xyzrho_ != NULL);
  for (i=0; i<natm_; i++) {
    xyzrho_[i*3+0] = xyzrho[i*3+0];
    xyzrho_[i*3+1] = xyzrho[i*3+1];
    xyzrho_[i*3+2] = xyzrho[i*3+2];
		//printf("Coordinate of rho atom %d %f %f %f\n", i, xyzrho_[i*3+0], xyzrho_[i*3+1], xyzrho_[i*3+2]);
  }
  inuc_ = inuc;
	//printf("Surface for atom %d\n", inuc_);
  xnuc_[0] = xyzrho[inuc_*3+0];
  xnuc_[1] = xyzrho[inuc_*3+1]; 
  xnuc_[2] = xyzrho[inuc_*3+2];
	//printf("Coordinate of rho %f %f %f\n", xnuc_[0], xnuc_[1], xnuc_[2]);
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
  occdrop_ = occdrop;
  nprims_ = nprims;
  nbas_= nbas;
  cart_ = cart;
  atm_ = atm;
  bas_ = bas;
  env_ = env;
  ao_loc_ = ao_loc;
	//non0tab_ = (int8_t *) malloc(sizeof(int8_t)*nbas_);
	non0tab_ = (unsigned char *) malloc(sizeof(unsigned char)*nbas_);
  assert(non0tab_ != NULL);
  for (i=0; i<nbas_; i++){
    non0tab_[i] = non0tab[i];
  }
	shls_ = (int *) malloc(sizeof(int)*2);
  assert(shls_ != NULL);
  shls_[0] = 0;
  shls_[1] = nbas_;

  // Crystal data
  nls_ = nls;
  nkpts_ = nkpts;
	//printf("The number cell vectors is %d\n", nls_);
	//printf("The number kpoints is %d\n", nkpts_);

	nmo_ = (int *) malloc(sizeof(int)*nkpts_);
  assert(nmo_ != NULL);
  for (i=0; i<nkpts_; i++){
	  nmo_[i] = 0;
	}
  for (i=0; i<nkpts_; i++){
  	for (j=0; j<nmo; j++){
    	if (fabs(mo_occ[i*nmo+j]) > occdrop_) nmo_[i] += 1;
		}
  }
  int max = nmo_[0];
	maxnmo_ = max;
  for (i=1; i<nkpts_; i++){
		if (nmo_[i] >= max) {
      max = nmo_[i];
			maxnmo_ = max;
    }
	}
	printf("Maximum occupied orbitals %d\n", maxnmo_);
  for (i=0; i<nkpts_; i++){
		printf("Number of occupied MO %d %d\n", i, nmo_[i]);
  }

	//For a 3D matrix L by N by M:
	// matrix[ i ][ j ][ k ] = array[ i*(N*M) + j*M + k ]
  //For a 2D matrix row*cols
	// maxtrix[i][j] = array[ i*cols + j ]
	mo_coeff_ = (double complex *) malloc(sizeof(double complex)*maxnmo_*nprims_*nkpts_);
  assert(mo_coeff_ != NULL);
	mo_occ_ = (double *) malloc(sizeof(double)*maxnmo_*nkpts_);
  assert(mo_occ_ != NULL);
  int l = 0;
	for (i=0; i<nkpts_; i++){
    //printf("coeff value %g + i%g\n",mo_coeff[i*(nprims*nmo)+0*(nmo)+0]);
    //printf("coeff value %g + i%g\n",mo_coeff[i*(nprims*nmo)+0*(nmo)+1]);
    //printf("coeff value %g + i%g\n",mo_coeff[i*(nprims*nmo)+0*(nmo)+2]);
		l = 0;
		for (k=0; k<nmo; k++){
      //printf("coeff value %g + i%g\n",mo_coeff[i*(nprims*nmo)+0*(nmo)+k]);
      if (fabs(mo_occ[i*nmo+k]) > occdrop_){
				mo_occ_[i*maxnmo_+l] = mo_occ[i*nmo+k];
	 	    for (j=0; j<nprims_; j++){
 					int idx1 = i*(nprims_*nmo)+j*(nmo)+k;
 					int idx2 = i*(nprims_*maxnmo_)+j*(maxnmo_)+l;
          mo_coeff_[idx2] = mo_coeff[idx1];
				}
				l += 1;
			}
		}
	}
	//exit(-1);
	//for (i=0; i<nkpts_; i++){
  //  printf("coeff value %g + i%g\n",mo_coeff_[i*(nprims*maxnmo_)+0*(maxnmo_)+0]);
  //  printf("coeff value %g + i%g\n",mo_coeff_[i*(nprims*maxnmo_)+0*(maxnmo_)+1]);
  //  printf("coeff value %g + i%g\n",mo_coeff_[i*(nprims*maxnmo_)+0*(maxnmo_)+2]);
	//}

	//for (i=0; i<nmo; i++){ // Orbital
  //  if (fabs(mo_occ[i]) > occdrop_){
  //    mo_occ_[l] = mo_occ[i];
	//	  for (j=0; j<nprims_; j++){
  //      mo_coeff_[l*nprims_+j] = mo_coeff[j*nprims_+i];
  //    }
  //    l += 1;
  //  }
	//}

	ls_ = (double *) malloc(sizeof(double)*nls_*3);
  assert(ls_ != NULL);
  for (i=0; i<nls_; i++){
    ls_[i*3+0] = ls[i*3+0];
    ls_[i*3+1] = ls[i*3+1];
    ls_[i*3+2] = ls[i*3+2];
	  //printf("Coordinate of ls %d %g %g %g\n", i,ls_[i*3+0], ls_[i*3+1], ls_[i*3+2]);
  }
	explk_ = (double complex *) malloc(sizeof(double complex)*nls_*nkpts_);
  assert(explk_ != NULL);
  for (i=0; i<nls_; i++){
  	for (j=0; j<nkpts_; j++){
    	explk_[i*nkpts_+j] = explk[i*nkpts_+j];
	    explk_[i*nkpts_+j] = explk[i*nkpts_+j];
	    explk_[i*nkpts_+j] = explk[i*nkpts_+j];
	    //printf("explk %d %g + i%g  %g + i%g  %g + i%g\n", i, explk_[i*nkpts_+j], explk_[i*nkpts_+j], explk_[i*nkpts_+j]);
		}
  }
	rcut_ = (double *) malloc(sizeof(double)*nbas_);
  assert(rcut_ != NULL);
  for (i=0; i<nbas_; i++){
    rcut_[i] = rcut[i];
	  //printf("rcut %d %g\n", i, rcut_[i]);
  }

	int ij = 0;
	double v[3];
	xyzrhoshell_ = (double *) malloc(sizeof(double)*nls_*natm_*3);
  assert(xyzrhoshell_ != NULL);
	idx_ = (int *) malloc(sizeof(int)*nls_*natm_);
  assert(idx_ != NULL);
  for (i=0; i<nls_; i++){
    v[0] = ls_[i*3+0];
    v[1] = ls_[i*3+1];
    v[2] = ls_[i*3+2];
	  //printf("Vector Shell %d %g %g %g\n", i,v[0],v[1],v[2]);
		for (j=0; j<natm_; j++){
			xyzrhoshell_[ij*3+0] = xyzrho_[j*3+0] + v[0];
			xyzrhoshell_[ij*3+1] = xyzrho_[j*3+1] + v[1];
			xyzrhoshell_[ij*3+2] = xyzrho_[j*3+2] + v[2];
			idx_[ij] = j;
	    //printf("Coordinate of shell %d atom %d %g %g %g\n", i, idx_[ij],xyzrhoshell_[ij*3+0], xyzrhoshell_[ij*3+1], xyzrhoshell_[ij*3+2]);
      ij += 1;
		}
  }

  double point[3], grad[3], rho, gradmod;
  point[0] = 0.0;
  point[1] = 0.0;
  point[2] = 0.0;
  rho_grad(point, &rho, grad, &gradmod);
	printf("Rhograd %f %f %f %f %f\n", rho, grad[0], grad[1], grad[2], gradmod);

  //surface();
	//for (i=0; i<npang_; i++){
  //  nlimsurf[i] = nlimsurf_[i];
	//  for (j=0; j<ntrial_; j++){
  //    rsurf[i*ntrial_+j] = rsurf_[i*ntrial_+j];
  //  }
  //}

  free(rcut_);
  free(ls_);
  free(explk_);
  free(mo_coeff_);
  free(mo_occ_);
  free(nmo_);
  free(coords_);
  free(xyzrho_);
  free(xyzrhoshell_);
  free(idx_);
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

inline void rho_grad(double *point, double *rho, double *grad, double *gradmod){

	double complex ao_[nprims_*4*nkpts_];
  double complex c0_[maxnmo_],c1_[maxnmo_],c2_[maxnmo_],c3_[maxnmo_];

  if (cart_ == 1) {
    aim_PBCGTOval_cart_deriv1(1, shls_, ao_loc_, ls_, nls_, explk_, nkpts_, ao_, 
                              point, rcut_, non0tab_, atm_, natm_, bas_, nbas_, env_);
  }
  else {
    aim_PBCGTOval_sph_deriv1(1, shls_, ao_loc_, ls_, nls_, explk_, nkpts_, ao_, 
                             point, rcut_, non0tab_, atm_, natm_, bas_, nbas_, env_);
  }

  int i, j, k;
  // ao[0] on kpoint 0
  for (k=0; k<nprims_; k++){
    //printf("%f +i%f \n", ao_[0*(4*nprims_)+0*nprims_+k]);
    //printf("%f +i%f \n", ao_[0*(4*nprims_)+1*nprims_+k]);
    //printf("%f +i%f \n", ao_[0*(4*nprims_)+2*nprims_+k]);
    //printf("%f +i%f \n", ao_[0*(4*nprims_)+3*nprims_+k]);
  }

  *rho = 0.0;
  grad[0] = 0.0;
  grad[1] = 0.0;
  grad[2] = 0.0;
  *gradmod = 0.0;

	for (k=0; k<nkpts_; k++){
  	for (i=0; i<nmo_[k]; i++){
	    c0_[i] = 0.0;
	    c1_[i] = 0.0;
	    c2_[i] = 0.0;
	    c3_[i] = 0.0;
	    for (j=0; j<nprims_; j++){
        int idx1 = k*(nprims_*maxnmo_)+j*(maxnmo_)+i;
	      c0_[i] += conj(ao_[k*(4*nprims_)+0*nprims_+j])*mo_coeff_[idx1];
	      c1_[i] += conj(ao_[k*(4*nprims_)+1*nprims_+j])*mo_coeff_[idx1];
	      c2_[i] += conj(ao_[k*(4*nprims_)+2*nprims_+j])*mo_coeff_[idx1];
	      c3_[i] += conj(ao_[k*(4*nprims_)+3*nprims_+j])*mo_coeff_[idx1];
	    }
	  }
	
	  for (i=0; i<nmo_[k]; i++){
	    *rho += conj(c0_[i])*c0_[i]*mo_occ_[k*maxnmo_+i];
	    grad[0] += conj(c1_[i])*c0_[i]*mo_occ_[k*maxnmo_+i]*2.0;
	    grad[1] += conj(c2_[i])*c0_[i]*mo_occ_[k*maxnmo_+i]*2.0;
	    grad[2] += conj(c3_[i])*c0_[i]*mo_occ_[k*maxnmo_+i]*2.0;
	  }
	}

  *rho *= 1.0/(double)nkpts_;
  grad[0] *= 1.0/(double)nkpts_;
  grad[1] *= 1.0/(double)nkpts_;
  grad[2] *= 1.0/(double)nkpts_;
  
  *gradmod = grad[0]*grad[0];
  *gradmod += grad[1]*grad[1];
  *gradmod += grad[2]*grad[2];
  *gradmod = sqrt(*gradmod);
  grad[0] *= 1.0/(*gradmod + HMINIMAL);
  grad[1] *= 1.0/(*gradmod + HMINIMAL);
  grad[2] *= 1.0/(*gradmod + HMINIMAL);

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
    ntrial_,epsroot_,rsurf_,nlimsurf_,rmaxsurf_,step_,epsilon_)
{
#pragma omp for schedule(dynamic) nowait
	for (i=0; i<npang_; i++){
    nintersec = 0;
    double cost = ct_[i];
    double sintcosp = st_[i]*cp_[i];
    double sintsinp = st_[i]*sp_[i];
    int ia = inuc_, ib;
    double ra = 0.0, rb;
	  for (j=0; j<ntrial_; j++){
      double ract = rpru_[j];
      xpoint[0] = xnuc_[0] + ract*sintcosp; 
      xpoint[1] = xnuc_[1] + ract*sintsinp; 
      xpoint[2] = xnuc_[2] + ract*cost;     
      //TODO: Better Check for error
      int ier = odeint(xpoint, step_, epsilon_);
      if (ier == 1) {
        cerror("too short steep on odeint");
      } else if (ier == 2) {
        cerror("too many steeps on odeint");
			} else if (ier == 4) {
        cerror("nna on odeint");
			}
      bool good = checkcp(xpoint, &ib);
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
        int ier = odeint(xpoint, step_, epsilon_);
      	if (ier == 1) {
        	cerror("too short steep on odeint");
	      } else if (ier == 2) {
        	cerror("too many steeps on odeint");
				} else if (ier == 4) {
        	cerror("nna on odeint");
				}
        bool good = checkcp(xpoint, &im);
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

//ier = 0 (correct), 1 (short step), 2 (too many iterations), 
//      3 (infty), 4 (nna), 5(undef)
int odeint(double *ystart, double h1, double eps){

	double rho, gradmod, hnext;
  double grad[3] = {0.0};
	int ier = 0, i, nuc;
  double dydx[3], y[3], yscal[3];

	rho_grad(ystart, &rho, grad, &gradmod);
  if (rho <= RHOEPS && gradmod <= GRADEPS){
    ier = 3;
    return ier;
  }
  
  double hmin = 0.0;
  double x1 = 0.0;
  double x2 = 1e40;
  double x = x1;
  double h = h1;
  y[0] = ystart[0];
  y[1] = ystart[1];
  y[2] = ystart[2];

	for (i=0; i<mstep_; i++){
	  rho_grad(y, &rho, dydx, &gradmod);
		yscal[0] = fmax(fabs(y[0]) + fabs(dydx[0]*h) + TINY, eps);
		yscal[1] = fmax(fabs(y[1]) + fabs(dydx[1]*h) + TINY, eps);
		yscal[2] = fmax(fabs(y[2]) + fabs(dydx[2]*h) + TINY, eps);
		if ((x+h-x2)*(x+h-x1) > 0.0) h = x2 - x;
		rkqs(y, dydx, &x, h, eps, yscal, &hnext);
		if ((x-x2)*(x2-x1) >= 0.0 || checkcp(y, &nuc)){
			ystart[0] = y[0];
			ystart[1] = y[1];
			ystart[2] = y[2];
      ier = 0;
			return ier;
		}
		if (fabs(hnext) <= hmin) cerror("Step size too small in odeint");
		if (i == (mstep_-1)) {
  	  printf("NNA at %f %f %f\n", y[0], y[1], y[2]);
	    cerror("Reached max steps in odeint");
		}
		h = hnext;
  }
    
	// Test if the point is far from RMAXSURF from current atom. 
  double a1 = y[0] - xnuc_[0];
  double a2 = y[1] - xnuc_[1];
  double a3 = y[2] - xnuc_[2];
  if ((a1*a1+a2*a2+a3*a3)>=5.0*5.0){
    ier = 3;
  } else { 
  	printf("NNA at %f %f %f\n", y[0], y[1], y[2]);
	  cerror("NNA found in odeint"); 
  }

  return ier;
}

void rkqs(double *y, double *dydx, double *x, 
          double htry, double eps,
	        double *yscal, double *hnext){
	
  int i;
	double yerr[3], ytemp[3], errmax, xnew, htemp;

	double h = htry;
  *hnext = 0.0;
  errmax = 0.0;

	for (;;){
    if (backend_ == 1) {
			steeper_rkck(y, dydx, h, ytemp, yerr);
		} else if (backend_ == 2) {
			steeper_rkdp(y, dydx, h, ytemp, yerr);
		}
		errmax = 0.0;
		for (i=0; i<3; i++) {
		  errmax = fmax(errmax, fabs(yerr[i]/yscal[i]));
    }
		errmax /= eps;
		if (errmax > 1.0) {
			htemp = SAFETY*h*pow(errmax, PSHRNK);
      h = fmin(fmax(fabs(htemp),0.1*fabs(h)),h);
			xnew = *x + h;
			if (xnew == *x) {
        cerror("stepsize underflow in rkqs");
      }
			continue;
		}
		else {
			if (errmax > ERRCON) {
				*hnext = SAFETY*h*pow(errmax, PGROW);
			} else {
				*hnext = 5.0*h;
			}
			*x += h;
			y[0] = ytemp[0];
			y[1] = ytemp[1];
			y[2] = ytemp[2];
			break; //return
		}
	}
}

inline void steeper_rkdp(double *xpoint, double *grdt, double h0, double *xout, double *xerr){

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

  static const double b1 = 5179.0/57600.0;
  static const double b3 = 7571.0/16695.0;
  static const double b4 = 393.0/640.0;
  static const double b5 = -92097.0/339200.0;
  static const double b6 = 187.0/2100.0;
  static const double b7 = 1.0/40.0;

  static double dc1 = c1-b1;
  static double dc3 = c3-b3;
  static double dc4 = c4-b4;
  static double dc5 = c5-b5;
  static double dc6 = c6-b6;
  static double dc7 = -b7;

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

inline void steeper_rkck(double *xpoint, double *grdt, double h0, double *xout, double *xerr){

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
  static double dc1 = c1-(2825.0/27648.0);
  static double dc3 = c3-(18575.0/48384.0);
  static double dc4 = c4-(13525.0/55296.0);
  static double dc5 = -277.0/14336.0;
  static double dc6 = c6-(1.0/4.0);
  
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


bool checkcp(double *x, int *nuc){

  int i,j,ij;
  bool iscp = false;
  double rho, grad[3], gradmod;

  *nuc = -2;
  rho_grad(x, &rho, grad, &gradmod);

  ij = 0;
  for (i=0; i<nls_; i++){
		for (j=0; j<natm_; j++){
			if (fabs(x[0]-xyzrhoshell_[ij*3+0]) < epsiscp_ && 
			    fabs(x[1]-xyzrhoshell_[ij*3+1]) < epsiscp_ && 
			    fabs(x[2]-xyzrhoshell_[ij*3+2]) < epsiscp_){  
				iscp = true;
			  *nuc = ij;
				return iscp;
			}
      ij += 1;
		}
  }

  // Put in the begining
  if (gradmod <= GRADEPS){
    iscp = true;
    if (rho <= RHOEPS){
      *nuc = -1;
    }
  } 

  return iscp;
  
}

void cerror(const char *text){
	fprintf(stderr,"Error %s\n", text);
	exit(1);
}

