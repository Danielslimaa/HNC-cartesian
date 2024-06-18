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

//g++ main_TSSP.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void){
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  P = 1 << 8;
  N = P / 1;
  inv_N2 = 1. / ((double)((P - 1) * (P - 1)));
  L = 5.;
  
  double h = 2. * L / (double)(2 * (P - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 10.;
  rho = 1.0;
  dt = 0.001;
  printf("N = %d, L = %1.0f, h = %1.6f, dk = %1.6f\n", N, L, h, dkx);
  printf("U = %1.2f, rho = %1.2f, dt = %1.4f\n", U, rho, dt);
  int max_threads = 8;//omp_get_max_threads() / 2; // 16 cores 
  printf("Maximum number of threads = %d\n", max_threads);
  omp_set_num_threads(max_threads);

  int fftw_init_threads(void);
  fftw_plan_with_nthreads(max_threads);

  double * x = (double *)fftw_malloc(sizeof(double) * P * P);
  double * y = (double *)fftw_malloc(sizeof(double) * P * P);
  double * kx = (double *)fftw_malloc(sizeof(double) * P * P);
  double * ky = (double *)fftw_malloc(sizeof(double) * P * P);
  double * k2 = (double *)fftw_malloc(sizeof(double) * P * P);
  double * V = (double *)fftw_malloc(sizeof(double) * P * P);
  double * g = (double *)fftw_malloc(sizeof(double) * P * P);
  double * S = (double *)fftw_malloc(sizeof(double) * P * P);
  double * new_S = (double *)fftw_malloc(sizeof(double) * P * P);
  double * omega = (double *)fftw_malloc(sizeof(double) * P * P);
  double * Lg = (double *)fftw_malloc(sizeof(double) * P * P);
  double * expAx = (double *)fftw_malloc(sizeof(double) * P * P); 
  double * expAy = (double *)fftw_malloc(sizeof(double) * P * P);
  double * expA = (double *)fftw_malloc(sizeof(double) * P * P);

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

  fftw_plan p_omega = fftw_plan_r2r_2d(N, N, omega, omega, FFTW_REDFT00, FFTW_REDFT00, flags);  
  fftw_plan p_S = fftw_plan_r2r_2d(N, N, S, S, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan p_g = fftw_plan_r2r_2d(N, N, g, g, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};
  int howmany = N;
  //1D FFT's in x-axis (columns)  
  //fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, -1, flags);
  fftw_plan p_x = fftw_plan_many_r2r(1, &N, howmany, g, NULL, howmany, 1, g, NULL, howmany, 1, kinds, flags);

  //1D FFT's in y-axis (rows)
  fftw_plan p_y = fftw_plan_many_r2r(1, &N, howmany, g, NULL, 1, howmany, g, NULL, 1, howmany, kinds, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"KN%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  geometry(x, y, kx, ky, k2);
  /* The potential:
  a) The gaussian potential: "GEM2"
  b) The dipolar potential used by Robert Zillinger: "Dipolar_Zillinger"
  c) The Rydberg potential: "Rydberg"
  d) The QC-hexagonal: "QC_hexagonal"
  e) The Qc-dodecagonal: "QC_dodecagonal"
  */
  potential_V(x, y, k2, V, "QC_dodecagonal");
  printer_field_transversal_view(x, y, V, "QC_dodecagonal.dat");
  potential_V(x, y, k2, V, "QC_hexagonal");
  printer_field_transversal_view(x, y, V, "QC_hexagonal.dat");
  potential_V(x, y, k2, V, "GEM2");
  preliminaries_TSSP(expAx, expAy, expA, k2);
  initialize_g_S(x, y, g, S);

  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {   
    compute_nonKinetic_step(p_S, p_omega, k2, S, g, omega, V); 
    compute_kinetic_2D(g, expA, p_g);
    compute_nonKinetic_step(p_S, p_omega, k2, S, g, omega, V);

 
    print_loop(x, y, k2, g, V, S, new_S, counter);
    counter += 1;
  }
  printf("\nThe computation has ended. Printing the fields.\n");
  printer_field(x, y, g, "g_full.dat");

  void fftw_cleanup_threads(void);
  fftw_destroy_plan(p_omega);
  fftw_destroy_plan(p_S);
  fftw_destroy_plan(p_g);
  fftw_destroy_plan(p_x);
  fftw_destroy_plan(p_y);
  fftw_free(x);
  fftw_free(y);
  fftw_free(kx);
  fftw_free(ky);
  fftw_free(k2);
  fftw_free(V);
  fftw_free(g);
  fftw_free(S);
  fftw_free(new_S);
  fftw_free(omega);
  fftw_free(Lg);
  fftw_free(expAx); 
  fftw_free(expAy);
  fftw_free(expA);
  fftw_cleanup();
}