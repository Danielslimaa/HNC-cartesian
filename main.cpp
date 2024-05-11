#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <fftw3.h>
#include "functions.h"

long int N;
double inv_N2;
double L; 
double dx, dy, dkx, dky;
double U;
double rho;
double dt;
//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{
  N = 1 << 10;
  inv_N2 = 1. / ((double)(N * N));

  L = 16.;
  
  double h = 2. * L / (double)(2 * (N - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 1;
  rho = 1;
  dt = 0.00001;
  printf("N = %d, L = %1.0f, h = %1.6f, dk = %1.6f\n", N, L, h, dkx);
  printf("U = %1.2f, rho = %1.2f, dt = %1.4f\n", U, rho, dt);
  int max_threads = omp_get_max_threads();
  printf("Maximum number of threads = %d\n", max_threads);
  omp_set_num_threads(max_threads);

  int fftw_init_threads(void);
  fftw_plan_with_nthreads(max_threads);

  double * x = new double[N * N];
  double * y = new double[N * N];
  double * kx = new double[N * N];
  double * ky = new double[N * N];
  double * k2 = new double[N * N];
  double * V = new double[N * N];
  double * g = new double[N * N];
  double * S = new double[N * N];
  double * new_S = new double[N * N];
  double * omega = new double[N * N];
  double * Lg = new double[N * N];

  unsigned flags;
  bool with_wisdom = true;
  if(with_wisdom)
  {
    flags = FFTW_WISDOM_ONLY;
    printf ("Importing wisdom\n");
    char import_buffer[200];
    sprintf(import_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_import_wisdom_from_filename(import_buffer);
  }
  else
  {
    fftw_set_timelimit(FFTW_NO_TIMELIMIT);
    flags = FFTW_PATIENT;
    printf("Setting FFT plans.\n");
  }

  fftw_plan g_to_S =  fftw_plan_r2r_2d(N, N, new_S, new_S, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan omega_to_omega =  fftw_plan_r2r_2d(N, N, omega, omega, FFTW_REDFT00, FFTW_REDFT00, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  initialize_g_S(g, S);
  geometry(x, y, kx, ky, k2);
  potential_V(x, y, V);  
 
  bool condition = true; 
  double error = 1.;
  long int counter = 1;
  while(condition)
  {
    laplace_finite_difference(g, Lg);
    compute_omega(omega_to_omega, k2, S, omega);
    update_g(Lg, V, omega, g);
    compute_S(g_to_S, g, new_S);
    if(counter%1 == 0 or counter == 1)
    {
      error = compute_error(new_S, S);
      printf("t = %ld, error = %1.4e\n", counter, error);
      if(counter > 1)
      {
        condition = (error > 1e-10);
        printer_field_transversal_view(x, y, V, "Vtransversal.dat");
        printer_field_transversal_view(x, y, S, "Stransversal.dat");
        printer_field_transversal_view(x, y, g, "gtransversal.dat");
      }
    }
    memcpy(S, new_S, N * N * sizeof(double));
    counter += 1;
  }
  


  printer_field(x, y, g, "g.dat");
  printer_field(x, y, S, "S.dat");
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  delete[] x;
  delete[] y;
  delete[] kx;
  delete[] ky;
  delete[] k2;
  delete[] V;
  delete[] g;
  delete[] S;
  delete[] new_S;
  delete[] omega;
  delete[] Lg;
  return 0;
}