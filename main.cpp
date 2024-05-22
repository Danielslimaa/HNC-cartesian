#include "functions.h"

//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  N = 1 << 10;
  inv_N2 = 1. / ((double)(N * N));

  L = 750.;
  
  double h = 2. * L / (double)(2 * (N - 1));
  dx = h;
  dy = h;
  
  double dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
  dkx = dk;
  dky = dk;

  U = 0.01;
  rho = 1.9;
  dt = 0.0001;
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
  double * V = new double[N * N];
  double * g = new double[N * N];
  double * S = new double[N * N];
  double * Vph = new double[N * N];
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

  fftw_plan omega_to_omega =  fftw_plan_r2r_2d(N, N, omega, omega, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan Vph_to_Vph =  fftw_plan_r2r_2d(N, N, Vph, Vph, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan g_to_g =  fftw_plan_r2r_2d(N, N, g, g, FFTW_REDFT00, FFTW_REDFT00, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    sprintf(export_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }

  initialize_g_S(x, y, g, S);
  memcpy(new_S, S, N * N * sizeof(double));
  geometry(x, y, kx, ky, k2);
  /* The potential:
  a) The gaussian potential: "GEM2"
  b) The dipolar potential used by Robert Zillinger: "Dipolar_Zillinger"
  c) The Rydberg potential: "Rydberg"
  d) The QC-hexagonal: "QC_hexagonal"
  e) The Qc-dodecagonal: "QC_dodecagonal"
  */

  potential_V(x, y, k2, V, "QC_hexagonal");
 
  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {
    compute_omega(omega_to_omega, k2, S, omega);
    compute_Vph(V, g, omega, Vph);
    update_S(Vph_to_Vph, k2, Vph, S);
    compute_g(g_to_g, S, g);
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;
  }
  printf("\nThe computation has ended. Printing the fields.\n");
  printer_field(x, y, g, "g_full.dat");
  printer_field(x, y, S, "S_full.dat");
  
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
  delete[] Vph;
  delete[] new_S;
  delete[] omega;
  delete[] Lg;
  return 0;
}