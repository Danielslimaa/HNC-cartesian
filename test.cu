#include <cstdio>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_kernel(int * a, int * b) {
    printf("GPU: Value of a: %d\n", *a);
    b[0] = *a;
}

int main() {
    int N = 100;
    int * ha = new int[N];
    for (int i = 0; i < N; i++)
    {
      ha[i] = i-1;
    }
    int * a, * b;
    cudaMalloc(&a, sizeof(int) * N);
    cudaMalloc(&b, sizeof(int) * 1);
    cudaMemcpy(a,ha, sizeof(int) * N, cudaMemcpyHostToDevice);
    delete[] ha;
    print_kernel<<<1, 1>>>(&a[91],b);
    int * hb = new int[1];
    cudaMemcpy(hb,b, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // Wait for kernel to finish
    printf("Value of a: %d\n", *hb);
    cudaFree(a);
    cudaFree(b);
    delete[] hb;
    return 0;
}