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

//g++ K.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void){
  N = 1 << 10;
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  N = 1 << 8;
  inv_N2 = 1. / ((double)(N * N));

  L = 5.;
  
  double h = 2. * L / (double)(2 * (N - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 10;
  rho = 1;
  dt = 0.0001;
  printf("N = %d, L = %1.0f, h = %1.6f, dk = %1.6f\n", N, L, h, dkx);
  printf("U = %1.2f, rho = %1.2f, dt = %1.4f\n", U, rho, dt);
  int max_threads = 8;//omp_get_max_threads() / 2; // 16 cores 
  printf("Maximum number of threads = %d\n", max_threads);
  omp_set_num_threads(max_threads);

  int fftw_init_threads(void);
  fftw_plan_with_nthreads(max_threads);

  double * x = new double[N * N];
  double * y = new double[N * N];
  double * kx = new double[N * N];
  double * ky = new double[N * N];
  double * k2 = new double[N * N];
  double * exp_k2 = new double[N * N];
  double * V = new double[N * N];
  double * g = new double[N * N];
  double * fft_sqrt_g = new double[N * N];
  double * S = new double[N * N];
  double * omega = new double[N * N];
  double * Lg = new double[N * N];
  double * pre_W = new double[N * N];
  double * W = new double[N * N];
  double * f = new double[N * N];

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

  fftw_plan omega_to_omega =  fftw_plan_r2r_2d(N, N, omega, omega, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan S_to_g =  fftw_plan_r2r_2d(N, N, g, g, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan W_to_W =  fftw_plan_r2r_2d(N, N, W, W, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan f_to_f =  fftw_plan_r2r_2d(N, N, f, f, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan g_to_sqrt_g =  fftw_plan_r2r_2d(N, N, g, fft_sqrt_g, FFTW_REDFT00, FFTW_REDFT00, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  geometry(x, y, kx, ky, k2);
  potential_V(x, y, V);
  read_file(x, y, g, "g_full.dat");
  read_file(x, y, S, "S_full.dat");
  compute_omega(omega_to_omega, k2, S, omega);
  compute_utils(pre_W, exp_k2, g, fft_sqrt_g, pre_W_to_pre_W, g_to_sqrt_g);

  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {
    compute_kinetic(exp_k2, f, f_to_f);
    compute_W_part(k2, S, g, fft_sqrt_g, pre_W, W, f, W_to_W);  
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      f[i] *= exp(-(V[i] + omega[i] + W[i]) * dt);
    }    
    compute_kinetic(exp_k2, f, f_to_f); 
    normalize_f(f);
    printer_loop_f(counter, k2, g, f, V, omega, pre_W, W_to_W, f_to_f);

    counter += 1;
  }
  printf("\nThe computation has ended. Printing the fields.");
  printer_field(x, y, g, "g_full.dat");

  void fftw_cleanup_threads(void);
  fftw_cleanup();
  delete[] x;
  delete[] y;
  delete[] kx;
  delete[] ky;
  delete[] k2;
  delete[] exp_k2;
  delete[] V;
  delete[] g;
  delete[] fft_sqrt_g;
  delete[] S;
  delete[] omega;
  delete[] Lg;
  delete[] pre_W;
  delete[] W;
  delete[] f;
}