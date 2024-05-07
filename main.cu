#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include "kernels.cuh"

int main(void){
  int h_N = 1 << 10;
  double h_L = 40;
  double h_h = h_L / h_N;
  double h_rho = 1;
  cudaMemcpyToSymbol(N, &h_N, sizeof(int), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(h, &h_h, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(L, &h_L, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(rho, &h_rho, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  printf("N = %d, h = %f, L = %f\n", h_N, h_h, h_L);

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

  double * x, * y, * U;

  cudaMalloc(&x, sizeof(double) * h_N * h_N);
  cudaMalloc(&y, sizeof(double) * h_N * h_N);
  cudaMalloc(&U, sizeof(double) * h_N * h_N);

  initialize_geometry<<<Blocks_N, ThreadsPerBlock_N>>>(x, y);
  cudaDeviceSynchronize();
  initialize_U<<<Blocks_N, ThreadsPerBlock_N>>>(x, y, U);
  cudaDeviceSynchronize();

  printer_vector(U, "U.dat", h_N);
  
  return 0;
  cudaDeviceReset();
}