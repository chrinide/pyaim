// TODO: move all calls to one call

#include <stdlib.h>
#include <math.h>
#include "gto/grid_ao_drv.h"
#include "gto/aim_gto.h"
#include "vhf/fblas.h"

#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

double CINTcommon_fac_sp(int l);

void aim_GTOeval_loop(void (*fiter)(), FPtr_eval feval, FPtr_exp fexp, double fac,
                  int ngrids, int param[], int *shls_slice, int *ao_loc,
                  double *ao, double *coord, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t Ngrids = ngrids;

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
        double *buf = malloc(sizeof(double) * BLKSIZE*(NPRIMAX*2+ncart));
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                aoff = ao_loc[ish] - ao_loc[sh0];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                bgrids = MIN(ngrids-ip, BLKSIZE);
                (*fiter)(feval, fexp, fac, nao, Ngrids, bgrids,
                         param, shloc+iloc, ao_loc, buf, ao+aoff*Ngrids+ip,
                         coord+ip, non0table+ib*nbas,
                         atm, natm, bas, nbas, env);
        }
        free(buf);
}

void aim_GTOeval_spinor_drv(FPtr_eval feval, FPtr_exp fexp, void (*c2s)(), double fac,
                        int ngrids, int param[], int *shls_slice, int *ao_loc,
                        double complex *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t Ngrids = ngrids;

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        int ncart = NCTR_CART * param[TENSOR] * param[POS_E1];
        double *buf = malloc(sizeof(double) * BLKSIZE*(NPRIMAX*2+ncart));
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                aoff = ao_loc[ish] - ao_loc[sh0];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                bgrids = MIN(ngrids-ip, BLKSIZE);
                GTOeval_spinor_iter(feval, fexp, c2s, fac,
                                    nao, Ngrids, bgrids,
                                    param, shloc+iloc, ao_loc, buf, ao+aoff*Ngrids+ip,
                                    coord+ip, non0table+ib*nbas,
                                    atm, natm, bas, nbas, env);
        }
        free(buf);
}


void aim_GTOeval_sph_drv(FPtr_eval feval, FPtr_exp fexp, double fac, int ngrids,
                     int param[], int *shls_slice, int *ao_loc,
                     double *ao, double *coord, char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        aim_GTOeval_loop(GTOeval_sph_iter, feval, fexp, fac, ngrids,
                     param, shls_slice, ao_loc,
                     ao, coord, non0table, atm, natm, bas, nbas, env);
}

void aim_GTOeval_cart_drv(FPtr_eval feval, FPtr_exp fexp, double fac, int ngrids,
                      int param[], int *shls_slice, int *ao_loc,
                      double *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        aim_GTOeval_loop(GTOeval_cart_iter, feval, fexp, fac, ngrids,
                     param, shls_slice, ao_loc,
                     ao, coord, non0table, atm, natm, bas, nbas, env);
}

void aim_GTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                        double *ao, double *coord, char *non0table,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        aim_GTOeval_cart_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1, 1,
                         ngrids, param, shls_slice, ao_loc,
                         ao, coord, non0table, atm, natm, bas, nbas, env);
}
void aim_GTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                       double *ao, double *coord, char *non0table,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        aim_GTOeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1, 1,
                        ngrids, param, shls_slice, ao_loc,
                        ao, coord, non0table, atm, natm, bas, nbas, env);
}

void aim_GTOval_spinor_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                          double complex *ao, double *coord, char *non0table,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        aim_GTOeval_spinor_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                           CINTc2s_ket_spinor_sf1, 1,
                           ngrids, param, shls_slice, ao_loc,
                           ao, coord, non0table, atm, natm, bas, nbas, env);
}

