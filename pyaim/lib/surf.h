/*
 * Atoms in moleucles module
 * Author: Jose Luis Casals Sainz <jluiscasalssainz@gmail.com>
 */

#include <float.h>
#define MAX_THREADS 256

// Atm and basis info
double *coord_;
int *atm_;
int natm_;
int *bas_;
int nbas_;
double *env_;
int cart_;
double *mo_coeff_;
double *mo_occ_;
int nprim_;
int nmo_;
int8_t *non0tab_;
int *shls_;
int *ao_loc_;

// Surface info
#define EPS 1e-7
#define GRADEPS 1e-10
#define RHOEPS 1e-10
#define MINSTEP 1e-4
#define MAXSTEP 0.75
#define SAFETY 0.8
#define HMINIMAL DBL_EPSILON
#define ENLARGE 1.2
int inuc_;        
double epsiscp_;
int ntrial_;      
int npang_;
double epsroot_;
double rmaxsurf_;
int backend_;
double epsilon_;
double step_;
int mstep_;
double xnuc_[3];
double *rpru_;
double *ct_;
double *st_;
double *cp_;
double *sp_;
int *nlimsurf_;
double *rsurf_;

// Functions
void rhograd(double *point, double *rho, double *grad, double *gradmod);
bool checkcp(const double *x, const double rho, const double gradmod, int *nuc);
void stepper_rkck(const double *xpoint, const double *grdt, const double h0, double *xout, double *xerr);
bool adaptive_stepper(double *x, const double *grad, double *h);
int odeint(double *xpoint, double *rho, double *gradmod);

// Drivers
void surface();
void surf_driver(int inuc, int npang, double *ct, double *st, 
                 double *cp, double *sp,  
                 int ntrial, double *rpru, double epsiscp,
                 double epsroot, double rmaxsurf, int backend,
                 double epsilon, double step, int mstep,
                 int cart, double *coord, const double *xyzrho,
                 int *atm, int natm, 
                 int *bas, int nbas, double *env, int nprim,
                 int *ao_loc,
                 double *mo_coeff, double *mo_occ, int *nlimsurf, double *rsurf);

// AO evaluators
void aim_GTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);
void aim_GTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);
