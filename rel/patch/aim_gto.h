
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t bgrids);

void GTOeval_cart_iter(FPtr_eval feval,  FPtr_exp fexp, double fac,
                       size_t nao, size_t ngrids, size_t bgrids,
                       int param[], int *shls_slice, int *ao_loc, double *buf,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env);

int GTOshloc_by_atom(int *shloc, int *shls_slice, int *ao_loc, int *atm, int *bas);

void GTOeval_sph_iter(FPtr_eval feval,  FPtr_exp fexp, double fac,
                      size_t nao, size_t ngrids, size_t bgrids,
                      int param[], int *shls_slice, int *ao_loc, double *buf,
                      double *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env);

void GTOeval_spinor_iter(FPtr_eval feval, FPtr_exp fexp, void (*c2s)(), double fac,
                         size_t nao, size_t ngrids, size_t bgrids,
                         int param[], int *shls_slice, int *ao_loc, double *buf,
                         double complex *ao, double *coord, char *non0table,
                         int *atm, int natm, int *bas, int nbas, double *env);
