/*
 * Atoms in moleucles module
 * Author: Jose Luis Casals Sainz <jluiscasalssainz@gmail.com>
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "surf.h"

void surf_driver(int inuc, int npang, double *ct, double *st, 
                 double *cp, double *sp,  
                 int ntrial, double *rpru, double epsiscp,
                 double epsroot, double rmaxsurf, int backend,
                 double epsilon, double step, int mstep,
                 int cart, double *coord, int *atm, int natm, 
                 int *bas, int nbas, double *env, int nprim,
                 int *ao_loc,
                 double *mo_coeff, double *mo_occ){

  int i, j;

  // Setup surface info
  inuc_ = inuc;
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
  xnuc_[0] = coord[inuc_*natm_+0];
  xnuc_[1] = coord[inuc_*natm_+1]; 
  xnuc_[2] = coord[inuc_*natm_+2];

  // Basis info
  nprim_ = nprim;
  natm_ = natm;
  nbas_= nbas;
  cart_ = cart;
  coord_ = coord;
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
  assert(c0_ != NULL);
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

}

double rho(double *point){

  if (cart_ == 1) {
    aim_GTOval_cart_deriv1(1, shls_, ao_loc_, ao_, point, 
                       non0tab_, atm_, natm_, bas_, nbas_, env_);
  }
  else {
    aim_GTOval_sph_deriv1(1, shls_, ao_loc_, ao_, point, 
                      non0tab_, atm_, natm_, bas_, nbas_, env_);
  }

  int i, j;
  //cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
  //rho = numpy.empty((4,ngrids))
  //c0 = numpy.dot(ao[0], cpos)
  //c1 = numpy.dot(ao[1], cpos)
  for (i=0; i<nmo_; i++){
    c0_[i] = 0.0;
    for (j=0; j<nprim_; j++){
      c0_[i] += ao_[j+nprim_*0]*mo_coeff_[i*nprim_+j];
    }
  }

  //rho[0] = numpy.einsum('pi,pi->p', c0, c0)
  //rho[1] = numpy.einsum('pi,pi->p', c0, c1) * 2
  double rhoval = 0.0;
  for (i=0; i<nmo_; i++){
    rhoval += c0_[i]*c0_[i]*mo_occ_[i];
  }

  return rhoval;

}

void print_info(){

  int i;

  printf("Follow info for surface_driver\n");
  printf("\n"); 
  printf("General info\n");
  printf("###############################################\n");
  printf("Machine precision : %e\n", HMINIMAL);
  printf("Number of atoms : %d\n", natm_);
  printf("Coordinates of atoms : \n");
	for (i=0; i<natm_; i++){
    printf("%8.5f %8.5f %8.5f\n", coord_[i*natm_], coord_[i*natm_+1], coord_[i*natm_+2]);
	}
  printf("\n"); 
  printf("Surface info\n");
  printf("###############################################\n");
  printf("Finding surface for atom : %d\n", inuc_);
  printf("Coordinates of atom : %8.5f %8.5f %8.5f\n", xnuc_[0], xnuc_[1], xnuc_[2]);
  printf("Gradient tolerance : %e\n", GRADEPS);
  printf("Angular points in the mesh : %d\n", npang_);
  printf("Max limit for surface : %8.5f\n", rmaxsurf_);
  printf("Number of initial steeps in r-mesh : %d\n", ntrial_);
  printf("R-mesh values \n"); 
	for (i=0; i<ntrial_; i++){
    printf("%8.5f ", rpru_[i]);
  }
  printf("\n"); 
  printf("Save distance for CP : %8.5f \n", epsiscp_);
  printf("Bisection preccision : %8.5f \n", epsroot_);
  printf("ODE solver : %d\n", backend_);
  printf("ODE preccison : %8.5f\n", epsilon_);
  printf("ODE first step : %8.5f\n", step_);
  printf("ODE maximum iters : %d\n", mstep_);
  printf("\n"); 
  printf("Basis info\n");
  printf("###############################################\n");
  printf("Cart or spherical coordiantes : %d\n", cart_);
  printf("Number of basis functions : %d\n", nbas_);
  printf("Location of each AO\n");
	for (i=0; i<nbas_; i++){
    printf("%d ", ao_loc_[i]);
  }
  printf("\n"); 
  printf("Number of primitive functions : %d\n", nprim_);
  printf("Number of occupied orbitals : %d\n", nmo_);
  printf("MO occupations : \n");
  printf("\n"); 
  printf("RHO eval info\n");
  printf("###############################################\n");
  double *point, rhoval;
	point = (double *) malloc(sizeof(double)*3);
  point[0] = 0.0;
  point[1] = 0.1; 
  point[2] = 1.0; 
  for (i=0; i<5810; i++){
		rhoval = rho(point);
  }
  printf("The value of rho is : %8.5f \n", rhoval);

}
