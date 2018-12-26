// Functions
bool checkcp(const double *x, const double rho, const double gradmod, int *nuc);
inline void stepper_rkck(const double *xpoint, const double *grdt, const double h0, double *xout, double *xerr);
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
                 int *bas, int nbas, double *env, int nprim, int nmo,
                 int *ao_loc,
                 double *mo_coeff, double *mo_occ, int *nlimsurf, double *rsurf);

// AO evaluators
void aim_GTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);
void aim_GTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);
