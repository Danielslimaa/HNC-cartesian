#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>
#include <fftw3.h>

long int N;
double L; 
double h;

//g++ main.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void)
{

  N = 1 << 8;
  L = 40.;
  h = L / (double)N;

  double * x, * y, * U, * g, * S, * Vph;
  x = (double * )fftw_malloc(N * N * sizeof(double));
  y = (double * )fftw_malloc(N * N * sizeof(double));
  U = (double * )fftw_malloc(N * N * sizeof(double));
  g = (double * )fftw_malloc(N * N * sizeof(double));
  S = (double * )fftw_malloc(N * N * sizeof(double));
  Vph = (double * )fftw_malloc(N * N * sizeof(double));

    
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      x[i * N + j] = (-L / 2) + (i - 1) * h;
      y[i * N + j] = (-L / 2) + (j - 1) * h;
    }
  }

  void fftw_cleanup_threads(void);
  fftw_cleanup();
  return 0;
}