#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "vhf/fblas.h"
#include "gto/grid_ao_drv.h"
#include "pbc/aim_gto.h"

#define IMGBLK          40
#define OF_CMPLX        2
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

void aim_PBCeval_loop(void (*fiter)(), FPtr_eval feval, FPtr_exp fexp,
                  int ngrids, int param[], int *shls_slice, int *ao_loc,
                  double *Ls, int nimgs, double complex *expLk, int nkpts,
                  double complex *ao, double *coord,
                  double *rcut, unsigned char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        int shloc[shls_slice[1]-shls_slice[0]+1];
        const int nshblk = GTOshloc_by_atom(shloc, shls_slice, ao_loc, atm, bas);
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        const size_t Ngrids = ngrids;

        int i;
        int di_max = 0;
        for (i = shls_slice[0]; i < shls_slice[1]; i++) {
                di_max = MAX(di_max, ao_loc[i+1] - ao_loc[i]);
        }

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];
        const size_t nao = ao_loc[sh1] - ao_loc[sh0];
        int ip, ib, k, iloc, ish;
        size_t aoff, bgrids;
        size_t bufsize =((nimgs*3 + NPRIMAX*2 +
                          nkpts *param[POS_E1]*param[TENSOR]*di_max * OF_CMPLX +
                          IMGBLK*param[POS_E1]*param[TENSOR]*di_max +
                          param[POS_E1]*param[TENSOR]*NCTR_CART) * BLKSIZE
                         + nkpts * IMGBLK * OF_CMPLX + nimgs);
        double *buf = malloc(sizeof(double) * bufsize);
        for (k = 0; k < nblk*nshblk; k++) {
                iloc = k / nblk;
                ish = shloc[iloc];
                ib = k - iloc * nblk;
                ip = ib * BLKSIZE;
                aoff = (ao_loc[ish] - ao_loc[sh0]) * Ngrids + ip;
                bgrids = MIN(ngrids-ip, BLKSIZE);
                (*fiter)(feval, fexp, nao, Ngrids, bgrids, aoff,
                         param, shloc+iloc, ao_loc, buf,
                         Ls, expLk, nimgs, nkpts, di_max,
                         ao, coord+ip, rcut, non0table+ib*nbas,
                         atm, natm, bas, nbas, env);
        }
        free(buf);
}

void aim_PBCeval_cart_drv(FPtr_eval feval, FPtr_exp fexp,
                      int ngrids, int param[], int *shls_slice, int *ao_loc,
                      double *Ls, int nimgs, double complex *expLk, int nkpts,
                      double complex *ao, double *coord,
                      double *rcut, unsigned char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        aim_PBCeval_loop(PBCeval_cart_iter, feval, fexp,
                     ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                     ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void aim_PBCeval_sph_drv(FPtr_eval feval, FPtr_exp fexp,
                     int ngrids, int param[], int *shls_slice, int *ao_loc,
                     double *Ls, int nimgs, double complex *expLk, int nkpts,
                     double complex *ao, double *coord,
                     double *rcut, unsigned char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        aim_PBCeval_loop(PBCeval_sph_iter, feval, fexp,
                     ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                     ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void aim_PBCGTOval_cart_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        aim_PBCeval_cart_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

void aim_PBCGTOval_sph_deriv1(int ngrids, int *shls_slice, int *ao_loc,
                           double *Ls, int nimgs, double complex *expLk, int nkpts,
                           double complex *ao, double *coord,
                           double *rcut, unsigned char *non0table,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int param[] = {1, 4};
        aim_PBCeval_sph_drv(GTOshell_eval_grid_cart_deriv1, GTOcontract_exp1,
                         ngrids, param, shls_slice, ao_loc, Ls, nimgs, expLk, nkpts,
                         ao, coord, rcut, non0table, atm, natm, bas, nbas, env);
}

