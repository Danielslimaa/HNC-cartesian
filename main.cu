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
  h_N = 1 << 8;
  double h_L = 10;
  double h_h = h_L / h_N;
  double h_rho = 1;
  double h_dx = h_L / h_N;
  double h_dy = h_dx;
  double h_dk = 5.0 / h_N;
  double h_dkx = h_dk;
  double h_dky = h_dkx;
  double h_dt;
  double U = 10.;

  h_dt = 0.001;
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

  double * k2, * V, * g, * S, *new_S, * omega, * Vph, * second_term;
  cudaMalloc(&k2, sizeof(double) * h_N * h_N);
  cudaMalloc(&V, sizeof(double) * h_N * h_N);
  cudaMalloc(&g, sizeof(double) * h_N * h_N);
  cudaMalloc(&S, sizeof(double) * h_N * h_N);
  cudaMalloc(&new_S, sizeof(double) * h_N * h_N);
  cudaMalloc(&omega, sizeof(double) * h_N * h_N);
  cudaMalloc(&Vph, sizeof(double) * h_N * h_N);
  cudaMalloc(&second_term, sizeof(double) * h_N * h_N);

  double * x = new double[h_N * h_N];
  double * y = new double[h_N * h_N];
  double * kx = new double[h_N * h_N];
  double * ky = new double[h_N * h_N];
  double * h_k2 = new double[h_N * h_N];

  #pragma omp parallel for
  for (int i = 0; i < h_N; i++)
  {
    for (int j = 0; j < h_N; j++)
    {
      x[i * h_N + j] = (0) + (i - 1) * h_dx;
      y[i * h_N + j] = (0) + (j - 1) * h_dy;
      kx[i * h_N + j] = (0) + (i - 1) * h_dkx;
      ky[i * h_N + j] = (0) + (j - 1) * h_dky;
      h_k2[i * h_N + j] = kx[i * h_N + j] * kx[i * h_N + j] + ky[i * h_N + j] * ky[i * h_N + j];
    }
  }
  cudaMemcpy(k2, h_k2, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  double * h_V = new double[h_N * h_N];
  double * tmp = new double[h_N * h_N];
  #pragma omp parallel for
  for (int i = 0; i < h_N * h_N; i++)
  {
    h_V[i] = U * exp( -x[i] * x[i] - y[i] * y[i] );
    tmp[i] = 1.0;
  }  
  cudaMemcpy(V, h_V, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  printer_vector(x, y, V, "U.dat", h_N);  


  cudaMemcpy(g, tmp, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  cudaMemcpy(S, tmp, sizeof(double) * h_N * h_N, cudaMemcpyHostToDevice);
  delete[] tmp;
  printer_vector(x, y, g, "g0.dat", h_N);

  numStreams = h_N; // Number of CUDA streams
  // Allocate memory for the array of streams
  cudaStream_t * streams_y = new cudaStream_t[numStreams];
  cudaStream_t * streams_x = new cudaStream_t[numStreams];
  cudaEvent_t * events_x = new cudaEvent_t[numStreams];
  cudaEvent_t * events_y = new cudaEvent_t[numStreams];
  // Create each stream
  printf("Creating streams and cudaEvents\n");
  int * h_index = new int[h_N];
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamCreate(&streams_y[i]));
    CUDA_CHECK(cudaStreamCreate(&streams_x[i]));
    CUDA_CHECK(cudaEventCreate(&events_x[i]));
    CUDA_CHECK(cudaEventCreate(&events_y[i]));
    h_index[i] = i;
  }  
  printf("Streams created.\n");
  int * index;
  cudaMalloc(&index, sizeof(int) * h_N);
  cudaMemcpy(index, h_index, sizeof(int) * h_N, cudaMemcpyHostToDevice);
  delete[] h_index;
  bool condition = true;
  long int counter = 1;
  while(counter < 4)
  {    
    compute_second_term(g, second_term, numBlocks, threadsPerBlock);
    compute_omega(omega, k2, g, S, events_x, events_y, streams_x, streams_y, numBlocks, threadsPerBlock, index);
    compute_Vph_k(V, second_term, g, omega, Vph, events_x, events_y, streams_x, streams_y, numBlocks, threadsPerBlock, index);
    update_S<<<Blocks_N, ThreadsPerBlock_N>>>(S, k2, Vph);
    IFFT_S2g(g, S, events_x, events_y, streams_x, streams_y, numBlocks, threadsPerBlock, index);
    printf("counter = %ld\n", counter);
    counter++;
  }  

  printer_vector(x, y, g, "g.dat", h_N);
  // Destroy each stream
  printf("Destroy each stream\n");
  for (int i = 0; i < numStreams; ++i) 
  {
    CUDA_CHECK(cudaStreamDestroy(streams_y[i]));
    CUDA_CHECK(cudaStreamDestroy(streams_x[i]));
    CUDA_CHECK(cudaEventDestroy(events_x[i]));
    CUDA_CHECK(cudaEventDestroy(events_y[i]));
  }
  delete[] streams_y;
  delete[] streams_x;
  
  delete[] x;
  delete[] y;
  delete[] h_V;
  delete[] h_k2;
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