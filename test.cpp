#include "functions.h"

//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  padded_N = 1 << 10;
  N = padded_N / 2;
  inv_N2 = 1. / ((double)(padded_N * padded_N));

  L = 5.;
  
  double h = 2. * L / (double)(2 * (N - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 10;
  rho = 1.0;
  dt = 0.001;
  printf("N = %d, L = %1.0f, h = %1.6f, dk = %1.6f\n", N, L, h, dkx);
  printf("U = %1.2f, rho = %1.2f, dt = %1.4f\n", U, rho, dt);
  int max_threads = 8;//omp_get_max_threads() / 2; // 16 cores 
  printf("Maximum number of threads = %d\n", max_threads);
  omp_set_num_threads(max_threads);

  int fftw_init_threads(void);
  fftw_plan_with_nthreads(max_threads);

  double * x = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * y = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * kx = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * ky = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * k2 = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * V = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * g = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * S = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * Vph = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * new_S = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * omega = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);
  double * Lg = (double *)fftw_malloc(sizeof(double) * padded_N * padded_N);

  memset(x, 0, sizeof(double) * padded_N * padded_N);
  memset(y, 0, sizeof(double) * padded_N * padded_N);
  memset(kx, 0, sizeof(double) * padded_N * padded_N);
  memset(ky, 0, sizeof(double) * padded_N * padded_N);
  memset(k2, 0, sizeof(double) * padded_N * padded_N);
  memset(V, 0, sizeof(double) * padded_N * padded_N);
  memset(g, 0, sizeof(double) * padded_N * padded_N);
  memset(S, 0, sizeof(double) * padded_N * padded_N);
  memset(Vph, 0, sizeof(double) * padded_N * padded_N);
  memset(new_S, 0, sizeof(double) * padded_N * padded_N);
  memset(omega, 0, sizeof(double) * padded_N * padded_N);
  memset(Lg, 0, sizeof(double) * padded_N * padded_N);

  fftw_plan p =  fftw_plan_r2r_2d(padded_N, padded_N, g, g, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);
 
  read_field(x, y, g, "g_full.dat");

  fftw_execute(p);
  printer_field_transversal_view(x, y, g, "gt.dat");
  fftw_execute(p);
  #pragma omp parallel for
  for(int i = 0; i < padded_N * padded_N; i++)
  {
    g[i] *= inv_N2 * 0.25;
  }
  printer_field(x, y, g, "g_full2.dat");
  
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  fftw_free(x);
  fftw_free(y);
  fftw_free(kx);
  fftw_free(ky);
  fftw_free(k2);
  fftw_free(V);
  fftw_free(g);
  fftw_free(S);
  fftw_free(Vph);
  fftw_free(new_S);
  fftw_free(omega);
  fftw_free(Lg);
  return 0;
}