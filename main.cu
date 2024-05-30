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

int main(void){
  int h_N = 1 << 8;
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

  double * U;
  cudaMalloc(&U, sizeof(double) * h_N * h_N);

  double * x = new double[h_N * h_N];
  double * y = new double[h_N * h_N];

  #pragma omp parallel for
  for (int i = 0; i < h_N; i++)
  {
    for (int j = 0; j < h_N; j++)
    {
      x[i * h_N + j] = (-h_L / 2.) + (i - 1) * h_h;
      y[i * h_N + j] = (-h_L / 2.) + (j - 1) * h_h;
    }
  }

  double * h_U = new double[h_N * h_N];
  #pragma omp parallel for
  for (int i = 0; i < h_N * h_N; i++)
  {
    h_U[i] = exp( -x[i] * x[i] - y[i] * y[i] );
  }  
  cudaMemcpy(U, h_U, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  printer_vector(x, y, U, "U.dat", h_N);  

  const int numStreams = N; // Number of CUDA streams
  // Allocate memory for the array of streams
  cudaStream_t* streams_y = new cudaStream_t[numStreams];
  cudaStream_t* streams_x = new cudaStream_t[numStreams];

  // Create each stream
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamCreate(&streams_y[i]));
    CUDA_CHECK(cudaStreamCreate(&streams_x[i]));
  }  


  dim3 threadsPerBlock(h_N, h_N);
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
  DCT_x<<<numBlocks, threadsPerBlock>>>(U, U);
  DCT_y<<<numBlocks, threadsPerBlock>>>(U, U);
  printer_vector(x, y, U, "FFT_U.dat", h_N);  
  DCT_x<<<numBlocks, threadsPerBlock>>>(U, U);
  DCT_y<<<numBlocks, threadsPerBlock>>>(U, U);
  rescaling<<<Blocks_N, ThreadsPerBlock_N>>>(U);
  printer_vector(x, y, U, "IFFT_FFT_U.dat", h_N);                              

  // Synchronize and destroy each stream
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamDestroy(streams_y[i]));
    CUDA_CHECK(cudaStreamDestroy(streams_x[i]));
  }

  cudaDeviceReset();
  delete[] x;
  delete[] y;
  delete[] h_U;
  delete[] streams_y;
  delete[] streams_x;
  return 0;
}