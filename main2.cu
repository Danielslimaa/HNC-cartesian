#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>
#include "kernels.cuh"

int main(void)
{
  h_N = 1 << 7;
  double h_L = 10;
  double h_h = h_L / h_N;
  double h_rho = 1;
  double h_dx = h_L / h_N;
  double h_dy = h_dx;
  double h_dk = M_PI / h_L;
  double h_dkx = h_dk;
  double h_dky = h_dkx;
  double h_dt;
  double U = 10.;

  h_dt = 0.01;
  cudaMemcpyToSymbol(N, &h_N, sizeof(int), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(h, &h_h, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(L, &h_L, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(rho, &h_rho, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dx, &h_dx, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dy, &h_dy, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dkx, &h_dkx, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dky, &h_dky, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dt, &h_dt, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  printf("N = %d, h = %f, L = %f\n", h_N, h_h, h_L);
  printer_constants<<<1,1>>>();

  int Blocks_N, ThreadsPerBlock_N;
  if (h_N * h_N >= 1024)
  {
    ThreadsPerBlock_N = 1024;
  }
  else
  {
    ThreadsPerBlock_N = h_N * h_N;
  }
  Blocks_N = (int)ceil((double)(h_N * h_N) / 1024.0);

  printf("Blocks_N = %d, ThreadsPerBlock_N = %d\n", Blocks_N, ThreadsPerBlock_N);

  double * x, * y, * kx, * ky, * k2, * V, * g, * S, *new_S, * omega, * Vph, * second_term;
  cudaMalloc(&x, sizeof(double) * h_N * h_N);
  cudaMalloc(&y, sizeof(double) * h_N * h_N);
  cudaMalloc(&kx, sizeof(double) * h_N * h_N);
  cudaMalloc(&ky, sizeof(double) * h_N * h_N);
  cudaMalloc(&k2, sizeof(double) * h_N * h_N);
  cudaMalloc(&V, sizeof(double) * h_N * h_N);
  cudaMalloc(&g, sizeof(double) * h_N * h_N);
  cudaMalloc(&S, sizeof(double) * h_N * h_N);
  cudaMalloc(&new_S, sizeof(double) * h_N * h_N);
  cudaMalloc(&omega, sizeof(double) * h_N * h_N);
  cudaMalloc(&Vph, sizeof(double) * h_N * h_N);
  cudaMalloc(&second_term, sizeof(double) * h_N * h_N);

  double * h_x = new double[h_N * h_N];
  double * h_y = new double[h_N * h_N];
  double * h_kx = new double[h_N * h_N];
  double * h_ky = new double[h_N * h_N];
  double * h_k2 = new double[h_N * h_N];

  #pragma omp parallel for
  for (int i = 0; i < h_N; i++)
  {
    for (int j = 0; j < h_N; j++)
    {
      h_x[i * h_N + j] = (0) + i * h_dx;
      h_y[i * h_N + j] = (0) + j * h_dy;
      h_kx[i * h_N + j] = (0) + i * h_dkx;
      h_ky[i * h_N + j] = (0) + j * h_dky;
      h_k2[i * h_N + j] = h_kx[i * h_N + j] * h_kx[i * h_N + j] + h_ky[i * h_N + j] * h_ky[i * h_N + j];
    }
  }
  cudaMemcpy(x, h_x, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(y, h_y, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(kx, h_kx, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(ky, h_ky, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(k2, h_k2, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  double * h_V = new double[h_N * h_N];
  double * tmp = new double[h_N * h_N];
  #pragma omp parallel for
  for (int i = 0; i < h_N * h_N; i++)
  {
    h_V[i] = U * exp( -h_x[i] * h_x[i] - h_y[i] * h_y[i] );
    tmp[i] = 1;//exp( -h_x[i] * h_x[i] - h_y[i] * h_y[i] );
  }  
  cudaMemcpy(V, h_V, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  printer_vector(h_x, h_y, V, "U.dat", h_N);  

  cudaMemcpy(g, tmp, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(S, tmp, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  delete[] tmp;
  printer_vector(h_x, h_y, g, "g0.dat", h_N);
  int * h_index = new int[h_N];
  for (int i = 0; i < h_N; ++i) 
  {
    h_index[i] = i;
  }

  int * index;
  cudaMalloc(&index, sizeof(int) * h_N);
  cudaMemcpy(index, h_index, sizeof(int) * h_N, cudaMemcpyHostToDevice);
  delete[] h_index;

  /*
  //FFT_x<<<Blocks_N, ThreadsPerBlock_N>>>(g, x, kx);
  cudaDeviceSynchronize();
  FFT_y<<<Blocks_N, ThreadsPerBlock_N>>>(g, y, ky);
  cudaDeviceSynchronize();
  //laplace<<<Blocks_N, ThreadsPerBlock_N>>>(k2,g);
  cudaDeviceSynchronize();
  //IFFT_x<<<Blocks_N, ThreadsPerBlock_N>>>(g, x, kx);
  cudaDeviceSynchronize();
  IFFT_y<<<Blocks_N, ThreadsPerBlock_N>>>(g, y, ky);  
  cudaDeviceSynchronize();
  */

  ffty_test<<<1, h_N>>>(S, g, &index[5]);
  printer_array(S, "S1.dat", h_N);
  iffty_test<<<1, h_N>>>(S, g, &index[5]);
  
  //IFFT_y<<<1, h_N>>>(g, y, ky, &index[2]);

  printer_array(g, "g.dat", h_N);
  printer_array(S, "S2.dat", h_N);
  //printer_vector(h_x, h_y, g, "g.dat", h_N);

  delete[] h_x;
  delete[] h_y;
  delete[] h_V;
  delete[] h_k2;
  cudaFree(x);
  cudaFree(y);
  cudaFree(kx);
  cudaFree(ky);
  cudaFree(k2);
  cudaFree(V);
  cudaFree(g);
  cudaFree(S);
  cudaFree(new_S);
  cudaFree(omega);
  cudaFree(Vph);
  cudaFree(second_term);
  cudaFree(index);
  cudaDeviceReset();
  return 0;
}