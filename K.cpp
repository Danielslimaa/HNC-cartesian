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
  N = 1 << 8;
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  N = 1 << 8;
  inv_N2 = 1. / ((double)(N * N));

  L = 2.;
  
  double h = 2. * L / (double)(2 * (N - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 1;
  rho = 2.50;
  dt = 0.001;
  printf("N = %d, L = %1.0f, h = %1.6f, dk = %1.6f\n", N, L, h, dkx);
  printf("U = %1.2f, rho = %1.2f, dt = %1.4f\n", U, rho, dt);
  int max_threads = 16;//omp_get_max_threads() / 2; // 16 cores 
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
  double * sqrt_g = new double[N * N];
  double * fft_sqrt_g = new double[N * N];
  //compute_W_part(fft_sqrt_g, W, f, W_to_W);  
  double * S = new double[N * N];
  double * omega = new double[N * N];
  double * Lg = new double[N * N];
  double * pre_W = new double[N * N];
  double * W = new double[N * N];
  double * f = new double[N * N];
  double * expAx = new double[N * N]; 
  double * expAy = new double[N * N];
  double * Wx = new double[N * N];
  double * Wy = new double[N * N];

  double p = dk;

  unsigned flags;
  bool with_wisdom = false;
  if(with_wisdom)
  {
    flags = FFTW_WISDOM_ONLY;
    printf ("Importing wisdom\n");
    char import_buffer[200];
    sprintf(import_buffer,"KN%d_fftw3.wisdom", N);
    int numbrerwisdom = fftw_import_wisdom_from_filename(import_buffer);
  }
  else
  {
    fftw_set_timelimit(FFTW_NO_TIMELIMIT);
    flags = FFTW_PATIENT;
    printf("Setting FFT plans.\n");
  }

  fftw_plan S_to_g = fftw_plan_r2r_2d(N, N, g, g, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan f_to_f = fftw_plan_r2r_2d(N, N, f, f, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan W_to_W = fftw_plan_r2r_2d(N, N, W, W, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};
  int howmany = N;
  //1D FFT's in x-axis (columns)  
  //fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, -1, flags);
  fftw_plan p_x = fftw_plan_many_r2r(1, &N, howmany, f, NULL, howmany, 1, f, NULL, howmany, 1, kinds, flags);

  //1D FFT's in y-axis (rows)
  fftw_plan p_y = fftw_plan_many_r2r(1, &N, howmany, f, NULL, 1, howmany, f, NULL, 1, howmany, kinds, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"KN%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  geometry(x, y, kx, ky, k2);
  potential_V(x, y, V, "Dipolar_Zillinger");
  read_field(x, y, g, "g_full.dat");
  read_field(x, y, S, "S_full.dat");
  preliminaries(k2, g, S, omega, W, Wx, Wy, sqrt_g, fft_sqrt_g);

  //Operator in the x-direction
  #pragma omp parallel for
  for (int j = 0; j < N; j++)
  {
    for (int i = 0; i < N; i++)
    {
      double temp = (   pow(p * i, 2.0) + rho * dx * dy * Wx[i] ) * dt / 2.0;
      expAx[i * N + j] = exp(-temp) / (1.0 * N);
    }    
  }

  //Operator in the y-direction
  for (int i = 0; i < N; i++)
  {    
    for (int j = 0; j < N; j++)
    {
      double temp = (  pow(p * j, 2) + rho * dx * dy * Wy[i]  ) * dt / 2.0;
      expAy[i * N + j] = exp(-temp) / (1.0 * N);
    }    
  }

  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    f[i] = g[i];
  }

  normalize_f(f);

  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {
    compute_part1(expAx, expAy, f, sqrt_g, p_x, p_y);
    //compute_W_part(fft_sqrt_g, W, f, W_to_W);  
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      f[i] *= exp(-(V[i] + omega[i]) * dt);
    }    
    compute_part1(expAx, expAy, f, sqrt_g, p_x, p_y);
    normalize_f(f);
    printer_loop_f(counter, k2, g, f, V, omega, pre_W, W_to_W, f_to_f);

    counter += 1;
  }
  printf("\nThe computation has ended. Printing the fields.\n");
  printer_field(x, y, f, "f_full.dat");

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
  delete[] sqrt_g;
  delete[] fft_sqrt_g;
  delete[] S;
  delete[] omega;
  delete[] Lg;
  delete[] pre_W;
  delete[] W;
  delete[] f;
  delete[] expAx; 
  delete[] expAy;
  delete[] Wx;
  delete[] Wy;
}