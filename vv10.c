#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double vv10(const int n1,
						const int n2,
            const double coef_C,
            const double coef_B,
            const double *coords11, 
            const double *rho11,
            const double *gnorm211,
            const double *weights11,
            const double *coords22, 
            const double *rho22,
            const double *gnorm222,
            const double *weights22){

  const double pi = 3.14159265358979323846264338328; 
  const double const43 = 4.0/3.0*pi;

  size_t idx, jdx;
  double vv10_e = 0.0;

  double coef_beta;
  coef_beta = 1.0/32.0*pow(3.0/(coef_B*coef_B),3.0/4.0);
  double kappa_pref;
  kappa_pref = coef_B*(1.5*pi)/pow(9.0*pi,1.0/6.0);

  for (idx=0; idx<n1; idx++){
    double point1x = coords11[idx*3+0];
    double point1y = coords11[idx*3+1];
    double point1z = coords11[idx*3+2];
    double rho1 = rho11[idx];
    double weigth1 = weights11[idx];
    double gamma1 = gnorm211[idx];
    double Wp1 = const43*rho1;
    double Wg1 = coef_C*pow(gamma1/(rho1*rho1),2);
    double W01 = sqrt(Wg1 + Wp1);
    double kappa1 = pow(rho1,1.0/6.0)*kappa_pref;
    double kernel = 0.0;
#pragma omp parallel default(none) reduction(+:kernel) \
    shared(point1x,point1y,point1z,coords22,rho22,weights22,kappa_pref,gnorm222,kappa1,W01)
{
#pragma omp for nowait schedule(dynamic)
    for (jdx=0; jdx<n2; jdx++){
      double R = (point1x-coords22[jdx*3+0])*(point1x-coords22[jdx*3+0]);
      R += (point1y-coords22[jdx*3+1])*(point1y-coords22[jdx*3+1]);
      R += (point1z-coords22[jdx*3+2])*(point1z-coords22[jdx*3+2]);
      double rho2 = rho22[jdx];
      double weigth2 = weights22[jdx];
      double gamma2 = gnorm222[jdx];
      double Wp2 = const43*rho2;
      double Wg2 = coef_C*pow(gamma2/(rho2*rho2),2);
      double W02 = sqrt(Wg2 + Wp2);
      double kappa2 = pow(rho2,1.0/6.0)*kappa_pref;
      double g = W01*R + kappa1;
      double gp = W02*R + kappa2;
      kernel += -1.5*weigth2*rho2/(g*gp*(g+gp));
    }
} 
    vv10_e += weigth1*rho1*(coef_beta + 0.5*kernel);
  }
 
  return vv10_e;

}
