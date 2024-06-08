#include "functions.h"

//g++ isotropic_main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3
//https://chatgpt.com/share/f3da4ffd-be86-4c8e-bce5-26ab1a528a66

int main(void)
{
  P = 1 << 11;
  N = P / 1;
  inv_N2 = 1. / ((double)((P - 1) * (P - 1)));
  L = 40.;
  
  double h = 2. * L / (double)(2 * (P - 1));
  dx = h;
  dy = h;

  dr = h;

  dk = h;
  
  //dk = 2. * M_PI / (2. * L); // dk = 2*pi/(2*L)
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

  double * x = new double[N];
  double * y = new double[N];
  double * kx = new double[N];
  double * ky = new double[N];
  double * k2 = new double[N];
  double * V = new double[N];
  double * g = new double[N];
  double * S = new double[N];
  double * Vph = new double[N];
  double * new_S = new double[N];
  double * omega = new double[N];
  double * Lg = new double[N];

  #pragma omp parallel for 
  for (int i = 0; i < N; i++)
  {
    g[i] = 1.0;
    S[i] = 1.0;
  }


  isotropic_compute_g(S, g);
  //printer_field2(g, "g0.dat");
  //printer_field2(S, "S0.dat");
  //printer_field_transversal_view(x, y, S, "Si.dat");
  //printer_field_transversal_view(x, y, g, "gi.dat");
  memcpy(new_S, S, N * sizeof(double));
  //printer_field2(S, "initial_new_S.dat");
  geometry(x, y, kx, ky, k2);
  /* The potential:
  a) The gaussian potential: "GEM2"
  b) The dipolar potential used by Robert Zillinger: "Dipolar_Zillinger"
  c) The Rydberg potential: "Rydberg"
  d) The QC-hexagonal: "QC_hexagonal"
  e) The Qc-dodecagonal: "QC_dodecagonal"
  */
  //potential_V(x, y, k2, V, "GEM2");
  

  #pragma omp parallel for 
  for (int i = 0; i < N; i++)
  {
    g[i] = 1.0;
    S[i] = 1.0;
    V[i] = exp(-pow(i * dr, 2));
  }


  printer_field2(V, "V.dat");
  condition = true; 
  tolerance = 1e-6;
  long int counter = 1;
  while(condition)
  {
    isotropic_compute_omega(k2, S, omega);
    //printer_field2(omega, "omega1.dat");
    isotropic_compute_Vph(V, g, omega, Vph);
    //printer_field2(Vph, "Vph1.dat");
    isotropic_update_S(k2, Vph, S);
    //printer_field2(S, "S1.dat");
    isotropic_compute_g(S, g);
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
*/
  printf("\nThe computation has ended. Printing the fields.\n");
  printer_field(x, y, g, "g_full.dat");
  printer_field(x, y, S, "S_full.dat");
  printer_field_transversal_view(x, y, g, "gf.dat");
  printer_field_transversal_view(x, y, S, "Sf.dat");

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
