#include <float.h>

// Atm and basis info
int natm_;
int nbas_;
int nmo_;
double occdrop_;
int nprims_;
int cart_;
double *__restrict__ coords_;
double *__restrict__ xyzrho_;
double *__restrict__ xyzrhoshell_;
int *__restrict__ idx_;
int *__restrict__ atm_;
int *__restrict__ bas_;
double *__restrict__ env_;
double *__restrict__ mo_coeff_;
double *__restrict__ mo_occ_;
//int8_t *__restrict__ non0tab_;
unsigned char *__restrict__ non0tab_;
int *__restrict__ shls_;
int *__restrict__ ao_loc_;
int nls_;
double *__restrict__ ls_;
int nkpts_;
double complex *__restrict__ explk_;
double *__restrict__ rcut_;
double a_[3][3];

// Surface info
#define EPS 1e-7
#define GRADEPS 1e-10
#define RHOEPS 1e-10
#define MINSTEP 1e-6
#define MAXSTEP 0.75
#define SAFETY 0.9
#define HMINIMAL DBL_EPSILON
#define ENLARGE 1.2
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
#define TINY 1.0e-30
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
double *__restrict__ rpru_;
double *__restrict__ ct_;
double *__restrict__ st_;
double *__restrict__ cp_;
double *__restrict__ sp_;
int *nlimsurf_;
double *rsurf_;

// Functions
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
                 const double occdrop,
                 const double *a,
                 const int nls,
                 const double *ls,
                 const int nkpts,
                 const double complex *explk,
                 const double *rcut,
                 const unsigned char *non0tab,
                 int *nlimsurf, double *rsurf);

void rho_grad(double *point, double *rho, double *grad, double *gradmod);
void surface(void);
bool checkcp(double *x, int *nuc);
void cerror(const char *text);

int odeint(double *ystart, double h1, double eps);
void rkqs(double *y, double *dydx, double *x, 
          double htry, double eps,
	        double *yscal, double *hnext);
void steeper_rkck(double *y, double *dydx, double h, double *yout, double *yerr);
void steeper_rkdp(double *y, double *dydx, double h, double *yout, double *yerr);

// AO evaluators
void aim_PBCGTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env);
void aim_PBCGTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env);

