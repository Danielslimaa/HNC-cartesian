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
  h_N = 1 << 8;
  double h_L = 5;
  double h_h = h_L / h_N;
  double h_rho = 1;
  double h_dx = h_L / h_N;
  double h_dy = h_dx;
  double h_dk = 13.0 / h_N;
  double h_dkx = h_dk;
  double h_dky = h_dkx;
  cudaMemcpyToSymbol(N, &h_N, sizeof(int), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(h, &h_h, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(L, &h_L, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(rho, &h_rho, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dx, &h_dx, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dy, &h_dy, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(dkx, &h_dkx, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol("dky", &h_dky, sizeof(double), size_t(0), cudaMemcpyHostToDevice);
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

  dim3 threadsPerBlock(h_N, h_N);
  dim3 numBlocks(h_N / threadsPerBlock.x, h_N / threadsPerBlock.y);

  printf("numBlocks = (%d, %d)\n", h_N / threadsPerBlock.x, h_N / threadsPerBlock.y);

  double * U, * g, * S;
  cudaMalloc(&U, sizeof(double) * h_N * h_N);
  cudaMalloc(&S, sizeof(double) * h_N * h_N);
  cudaMalloc(&g, sizeof(double) * h_N * h_N);

  double * x = new double[h_N * h_N];
  double * y = new double[h_N * h_N];
  double * kx = new double[h_N * h_N];
  double * ky = new double[h_N * h_N];

  #pragma omp parallel for
  for (int i = 0; i < h_N; i++)
  {
    for (int j = 0; j < h_N; j++)
    {
      x[i * h_N + j] = (0) + (i - 1) * h_dx;
      y[i * h_N + j] = (0) + (j - 1) * h_dy;
      kx[i * h_N + j] = (0) + (i - 1) * h_dkx;
      ky[i * h_N + j] = (0) + (j - 1) * h_dky;
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

  cudaMemcpy(g, h_U, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(S, h_U, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);

  printer_vector(x, y, g, "g0.dat", h_N);

  numStreams = h_N; // Number of CUDA streams
  // Allocate memory for the array of streams
  cudaStream_t * streams_y = new cudaStream_t[numStreams];
  cudaStream_t * streams_x = new cudaStream_t[numStreams];
  // Create each stream
  printf("Create each stream\n");
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamCreate(&streams_y[i]));
    CUDA_CHECK(cudaStreamCreate(&streams_x[i]));
  }  
  printf("Streams created.\n");
  
  FFT(g, S, streams_x, streams_y, numBlocks, threadsPerBlock);


  printer_vector(x, y, g, "g2.dat", h_N);
  // Destroy each stream
  printf("Destroy each stream\n");
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamDestroy(streams_y[i]));
    CUDA_CHECK(cudaStreamDestroy(streams_x[i]));
  }
  
  delete[] x;
  delete[] y;
  delete[] h_U;
  cudaFree(g);
  cudaFree(S);
  cudaFree(index);
  delete[] streams_y;
  delete[] streams_x;
  cudaDeviceReset();
  return 0;
}