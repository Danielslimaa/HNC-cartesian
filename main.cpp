#include "functions.h"

//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3
//https://chatgpt.com/share/f3da4ffd-be86-4c8e-bce5-26ab1a528a66

int main(void)
{
  void fftw_cleanup_threads(void);
  fftw_cleanup();
  P = 1 << 9;
  N = P / 1;
  inv_N2 = 1. / ((double)((P - 1) * (P - 1)));
  L = 10.;
  
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
  int max_threads = 16;//omp_get_max_threads() / 2; // 16 cores 
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
  double * Vph = (double *)fftw_malloc(sizeof(double) * P * P);
  double * new_S = (double *)fftw_malloc(sizeof(double) * P * P);
  double * omega = (double *)fftw_malloc(sizeof(double) * P * P);
  double * Lg = (double *)fftw_malloc(sizeof(double) * P * P);

  memset(x, 0, sizeof(double) * P * P);
  memset(y, 0, sizeof(double) * P * P);
  memset(kx, 0, sizeof(double) * P * P);
  memset(ky, 0, sizeof(double) * P * P);
  memset(k2, 0, sizeof(double) * P * P);
  memset(V, 0, sizeof(double) * P * P);
  memset(g, 0, sizeof(double) * P * P);
  memset(S, 0, sizeof(double) * P * P);
  memset(Vph, 0, sizeof(double) * P * P);
  memset(new_S, 0, sizeof(double) * P * P);
  memset(omega, 0, sizeof(double) * P * P);
  memset(Lg, 0, sizeof(double) * P * P);

  unsigned flags;
  bool with_wisdom = false;
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

  fftw_plan omega_to_omega =  fftw_plan_r2r_2d(P, P, omega, omega, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan Vph_to_Vph =  fftw_plan_r2r_2d(P, P, Vph, Vph, FFTW_REDFT00, FFTW_REDFT00, flags);
  fftw_plan g_to_g =  fftw_plan_r2r_2d(P, P, g, g, FFTW_REDFT00, FFTW_REDFT00, flags);

  if (!with_wisdom)
  {
    char export_buffer[200];
    printf(export_buffer,"N%d_fftw3.wisdom", N);
    int numberwisdom = fftw_export_wisdom_to_filename(export_buffer);
  }
  initialize_g_S(x, y, g, S);
  
  compute_g(g_to_g, S, g);
  //printer_field2(g, "g0.dat");
  //printer_field2(S, "S0.dat");
  //printer_field_transversal_view(x, y, S, "Si.dat");
  //printer_field_transversal_view(x, y, g, "gi.dat");
  memcpy(new_S, S, P * P * sizeof(double));
  //printer_field2(S, "initial_new_S.dat");
  geometry(x, y, kx, ky, k2);
  /* The potential:
  a) The gaussian potential: "GEM2"
  b) The dipolar potential used by Robert Zillinger: "Dipolar_Zillinger"
  c) The Rydberg potential: "Rydberg"
  d) The QC-hexagonal: "QC_hexagonal"
  e) The Qc-dodecagonal: "QC_dodecagonal"
  */
  potential_V(x, y, k2, V, "GEM2");
  
  printer_field2(V, "V.dat");
  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {
    compute_omega(omega_to_omega, k2, S, omega);
    //printer_field2(omega, "omega1.dat");
    compute_Vph(V, g, omega, Vph);
    //printer_field2(Vph, "Vph1.dat");
    update_S(Vph_to_Vph, k2, Vph, S);
    //printer_field2(S, "S1.dat");
    compute_g(g_to_g, S, g);
    //printer_field2(g, "g1.dat");
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;
  }
/*
    compute_omega(omega_to_omega, k2, S, omega);
    printer_field2(omega, "omega2.dat");
    compute_Vph(V, g, omega, Vph);
    printer_field2(Vph, "Vph2.dat");
    update_S(Vph_to_Vph, k2, Vph, S);
    printer_field2(S, "S2.dat");
    compute_g(g_to_g, S, g);
    printer_field2(g, "g2.dat");
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;


    compute_omega(omega_to_omega, k2, S, omega);
    printer_field2(omega, "omega3.dat");
    compute_Vph(V, g, omega, Vph);
    printer_field2(Vph, "Vph3.dat");
    update_S(Vph_to_Vph, k2, Vph, S);
    printer_field2(S, "S3.dat");
    compute_g(g_to_g, S, g);
    printer_field2(g, "g3.dat");
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;    

    compute_omega(omega_to_omega, k2, S, omega);
    printer_field2(omega, "omega4.dat");
    compute_Vph(V, g, omega, Vph);
    printer_field2(Vph, "Vph4.dat");
    update_S(Vph_to_Vph, k2, Vph, S);
    printer_field2(S, "S4.dat");
    compute_g(g_to_g, S, g);
    printer_field2(g, "g4.dat");
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;

    compute_omega(omega_to_omega, k2, S, omega);
    printer_field2(omega, "omega5.dat");
    compute_Vph(V, g, omega, Vph);
    printer_field2(Vph, "Vph5.dat");
    update_S(Vph_to_Vph, k2, Vph, S);
    printer_field2(S, "S5.dat");
    compute_g(g_to_g, S, g);
    printer_field2(g, "g5.dat");
    print_loop(x, y, k2, g, V, S, new_S, counter);    
    counter += 1;
*/
  printf("\nThe computation has ended. Printing the fields.\n");
  printer_field(x, y, g, "g_full.dat");
  printer_field(x, y, S, "S_full.dat");
  printer_field_transversal_view(x, y, g, "gf.dat");
  printer_field_transversal_view(x, y, S, "Sf.dat");
  void fftw_cleanup_threads(void);

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
  fftw_cleanup();
  return 0;
}  
