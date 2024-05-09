#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>
#include <fftw3.h>
#include "functions.h"

long int N;
double inv_N2;
double L; 
double h;
double U;
double rho;
double dt;
//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{
  N = 1 << 8;
  L = 40.;
  h = L / (double)N;

  U = 10;
  rho = 1;

  double * x, * y, * kx, * ky, * k2, * V, * g, * S, * Vph;
  x = (double * )fftw_malloc(N * N * sizeof(double));
  y = (double * )fftw_malloc(N * N * sizeof(double));
  kx = (double * )fftw_malloc(N * N * sizeof(double));
  ky = (double * )fftw_malloc(N * N * sizeof(double));
  k2 = (double * )fftw_malloc(N * N * sizeof(double));
  V = (double * )fftw_malloc(N * N * sizeof(double));
  g = (double * )fftw_malloc(N * N * sizeof(double));
  S = (double * )fftw_malloc(N * N * sizeof(double));
  Vph = (double * )fftw_malloc(N * N * sizeof(double));

  geometry(x, y, kx, ky, k2);
  potential_U(x, y, V);  
  printer_vector(x, y, V, "V.dat");

  bool condition = true; 

  while(condition)
  {
    
  }

  void fftw_cleanup_threads(void);
  fftw_cleanup();
  return 0;
}