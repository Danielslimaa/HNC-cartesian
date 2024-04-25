#include <fftw3.h>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#define REAL 0
#define IMAG 1

int N;
int inv_N2;
double L;
double h;
double U;
double rho;
double dt;

#include "functions.h"

//g++ hnc.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{
  N = 256;
  inv_N2 = 1.0 / (double)(N * N);
  L = 80.0;
  h = L / (double)N;
  dt = 0.01;
  U = 10.0;
  rho = 1.0;

  int max_threads = 8;//omp_get_max_threads();
  printf("Maximum number of threads = %d\n", max_threads);
  omp_set_num_threads(max_threads);

  int fftw_init_threads(void);
  fftw_plan_with_nthreads(max_threads);

  double * x = new double[N * N], * y = new double[N * N], * mu2_p = new double[N * N], * lambda2_q = new double[N * N], 
  * kx = new double[N * N], * ky = new double[N * N], * K = new double[N * N];
  printf("Computing the geometry\n");
  #pragma omp parallel for
  for(int i = 0; i < N; i++)
  {    
    for (int j = 0; j <= N / 2 - 1; j++)
    {
      ky[i * N + j] = j * 2 * M_PI / L;
    }    
    ky[i * N + (N / 2)] = 0;    
    for (int j = N / 2 + 1; j < N; j++)
    {
      ky[i * N + j] = (j - N) * 2 * M_PI / L;
    }
  }
  #pragma omp parallel for    
  for(int j = 0; j < N; j++)
  {
    for (int i = 0; i <= N / 2 - 1; i++)
    {
      kx[i * N + j] = i * 2 * M_PI / L;
    }
    kx[(N / 2) * N + j] = 0;
    for (int i = N / 2 + 1; i < N; i++)
    {
      kx[i * N + j] = (i - N) * 2.0 * M_PI / L;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      x[i * N + j] = (-L / 2) + (i - 1) * h;
      y[i * N + j] = (-L / 2) + (j - 1) * h;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N / 2; j++)
    {
      mu2_p[i * N + j] = pow(i * 2 * M_PI / L, 2.0);
    }    
    for (int j = N / 2; j < N; j++)
    {
      mu2_p[i * N + j] = pow((i - N) * 2 * M_PI / L, 2.0);
    }
  }
  
  #pragma omp parallel for
  for (int j = 0; j < N; j++)
  {    
    for (int i = 0; i < N / 2; i++)
    {
      lambda2_q[i * N + j] = pow(j * 2.0 * M_PI / L, 2.0);
    }    
    for (int i = N / 2; i < N; i++)
    {
      lambda2_q[i * N + j] = pow((j - N) * 2.0 * M_PI / L, 2.0);
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    K[i] = sqrt(mu2_p[i] + lambda2_q[i]);
  }

  fftw_complex * g, * new_g, * S, * omega, * omega_k, * V_ph;

  g = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
  new_g = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
  S = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
  omega = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
  omega_k = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);
  V_ph = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * N);

  fftw_plan S_to_g, omega_k_to_omega, Vph_to_Vph_k;

  unsigned flags;

  bool with_wisdom = true;
  if(with_wisdom == true)
  {
    flags = FFTW_WISDOM_ONLY;
    printf ("Importing wisdom \n");
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

  S_to_g = fftw_plan_dft_2d(N, N, S, g, +1, flags); 
  omega_k_to_omega = fftw_plan_dft_2d(N, N, omega, omega, +1, flags); 
  Vph_to_Vph_k = fftw_plan_dft_2d(N, N, V_ph, V_ph, -1, flags); 

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  double * V = new double[N * N];
  potential_U(x, y, V);

  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    g[i][0] = 1.0;
    S[i][0] = 1.0;
  }
  //memcpy(S, g, sizeof(fftw_complex) * N * N);

  dt = 0;
  double error;
  bool condition = true;
  long int t = 1;
  while(condition)
  {
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      omega[i][0] = - 0.25 * (mu2_p[i] + lambda2_q[i]) * (2.0 * S[i][0] + 1.0) * (1.0 - 1.0 / S[i][0]) * (1.0 - 1.0 / S[i][0]);
    }
    fftw_execute(omega_k_to_omega);

    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      V_ph[i][0] = rho * (g[i][0] * V[i]  + (g[i][0] - 1.0) * omega[i][0]) * inv_N2;
    }
    fftw_execute(Vph_to_Vph_k);

    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      S[i][0] = (1.0 - dt) * S[i][0] + dt * K[i] / sqrt(mu2_p[i] + lambda2_q[i] + 4.0 * V_ph[i][0]);
    }    

    #pragma omp parallel for
    for(int i = 0; i < N * N; i++)
    {
      S[i][0] = (S[i][0] - 1.0) * inv_N2 / (2.0 * M_PI * 2.0 * M_PI * rho);
    } 
    fftw_execute(S_to_g);

    #pragma omp parallel for
    for(int i = 0; i < N * N; i++)
    {
      new_g[i][0] = 1.0 + g[i][0];
    }

    if (t%100 == 0 or t == 1)
    {
      error = compute_error(new_g, g);
      printf("\rt = %d, error = %1.4e", t, error);
      condition = (error < 1e-6);
    }
    memcpy(g, new_g, sizeof(fftw_complex) * N * N);
    t = t + 1;
  }
  
  delete[] x; delete[] y; delete[] mu2_p; delete[] lambda2_q; delete[] kx; delete[] ky;
  delete[] K;
  delete[] V;
  void fftw_cleanup_threads(void);
  fftw_cleanup();

  
  return 0;
}