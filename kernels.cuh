#pragma once
#define CUDA_GRID_STRIDE_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < n; i += blockDim.x * gridDim.x)

#define MAX_BLKSZ 1024
#define WARPSZ 32
#define BLOCK_SIZE 32

#define PI M_PI

// Check for CUDA errors
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

__constant__ int N;
__constant__ double h;
__constant__ double L;
__constant__ double rho;
__constant__ double dx;
__constant__ double dy;
__constant__ double dkx;
__constant__ double dky;
__constant__ double dt;
int numStreams;
int h_N;

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ double Warp_sum(double var)
{
    unsigned mask = 0xffffffff;

    for (int diff = warpSize / 2; diff > 0; diff = diff / 2)
        var += __shfl_down_sync(mask, var, diff, 32);
    return var;
} /* Warp_sum */

__device__ double Shared_mem_sum(double shared_vals[])
{
    int my_lane = threadIdx.x % warpSize;

    for (int diff = warpSize / 2; diff > 0; diff = diff / 2)
    {
        /* Make sure 0 <= source < warpSize  */
        int source = (my_lane + diff) % warpSize;
        shared_vals[my_lane] += shared_vals[source];
    }
    return shared_vals[my_lane];
}

__global__ void DCT_x(
	double *__restrict__ X,
	double *__restrict__ Y
    )
{
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;

	double x_shfl_src, x_shfl_dest;

	double summation = 0.0;
    int n = 2 + N / 2;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x;  j < n; j += blockDim.x * gridDim.x)
    {
        #pragma unroll
        for (unsigned int m = 0; m < ((n - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
        {
            if ((m * BLOCK_SIZE + threadIdx.y) < n-2)
            {
                x_shfl_src = X[(threadIdx.y + m * BLOCK_SIZE) * N + j];
            }
            else
            {
                x_shfl_src = 0.0;
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 32; ++e)
            {
                // --- Column-major ordering - faster
                x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
                // y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
                summation += cos(PI * double(tid) * double(e + BLOCK_SIZE * m) / (n - 1)) * x_shfl_dest;
                // --- Row-major ordering - slower
                // y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
            }

            __syncthreads();
        }

        if (tid < n)
        {
            Y[tid * N + j] = rho * (X[tid * N + j] + pow(-1, tid) * X[(n-2) * N + j] + 2.0 * summation);
        }
    }
}

__global__ void DCT_y(
	double *__restrict__ X,
	double *__restrict__ Y
    )
{
    int n = 2 + N / 2;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double summation = 0.0;

    for (int i = blockIdx.y * blockDim.y + threadIdx.y;  i < n; i += blockDim.y * gridDim.y)
    {
        #pragma unroll
        for (unsigned int m = 0; m < ((n - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
        {
            if ((m * BLOCK_SIZE + threadIdx.y) < n-2)
            {
                x_shfl_src = X[i * N + threadIdx.y + m * BLOCK_SIZE];
            }
            else
            {
                x_shfl_src = 0.0;
            }
            __syncthreads();
            #pragma unroll
            for (int e = 0; e < 32; ++e)
            {
                // --- Column-major ordering - faster
                x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
                // y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
                summation += cos(PI * double(tid) * double(e + BLOCK_SIZE * m) / (n - 1)) * x_shfl_dest;
                // --- Row-major ordering - slower
                // y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
            }

            __syncthreads();
        }

        if (tid < n)
        {
            Y[i * N + tid] = rho * (X[i * N + tid] + pow(-1, tid) * X[i * N + n-2] + 2.0 * summation);
        }
    }
}

void printer_vector(double * x, double * y, double *vetor, const char *name, int h_N)
{
	double * h_vetor = new double[h_N * h_N];
	cudaMemcpy(h_vetor, vetor, sizeof(double) * h_N * h_N, cudaMemcpyDeviceToHost);
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < h_N * h_N; ++i)
	{
		myfile << x[i] << " " << y[i] << " " << h_vetor[i] << "\n";
        //std::cout << h_x[i] << "," << h_y[i] << "," << h_vetor[i] << std::endl;
	}
	myfile.close();
	delete[] h_vetor;
	return;
}

__global__ void initialize_geometry(double * x, double * y)
{
    CUDA_GRID_STRIDE_LOOP(i, N)
    {
        for (int j = 0; j < N; j++)
        {
            x[i * N + j] = -L / 2 + i * h;
            y[i * N + j] = -L / 2 + j * h;
        }
    }
}

__global__ void initialize_U(double * x, double * y, double * U)
{
    CUDA_GRID_STRIDE_LOOP(i, N * N)
    {
        U[i] = exp(-x[i] * x[i] - y[i] * y[i]);
    }
}

__global__ void rescaling(double * U)
{
    CUDA_GRID_STRIDE_LOOP(i, N * N)
    {
        U[i] /= (double)(N * N);
    }
}

/*
__global__ void g_from_S3(
	double *__restrict__ g,
	const double *__restrict__ S,
	double *sum_diff_g)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	double x_shfl_src, x_shfl_dest;

	double y_val = 0.0;

	#pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = S[threadIdx.x + m * BLOCK_SIZE] - 1.0;
			x_shfl_src *= (double)(threadIdx.x + m * BLOCK_SIZE);
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += j0f(double(tid) * dr * double(e + BLOCK_SIZE * m) * dk) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		sum_diff_g[0] += fabs(1.0 + (y_val * dk * dk / (2.0 * M_PI * rho)) - g[tid]);
		g[tid] = g[tid] * exp(dt * (1.0 + (y_val * dk * dk / (2.0 * M_PI * rho)) - g[tid]));
	}
}
*/
__global__ void ifft_cossine_x_integral(
	double *__restrict__ g,
	const double *__restrict__ S,
  int * index)
{
  int j = * index;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;

  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = S[(threadIdx.x + m * BLOCK_SIZE) * N + j] - 1.0;
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//#pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dx * double(e + BLOCK_SIZE * m) * dkx) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		g[tid] = y_val;
	}
}

__global__ void ifft_cossine_y_integral(
	double *__restrict__ g,
  int * index)
{
  int i = *index;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
	double c = dkx * dky / (2.0 * M_PI * 2.0 * M_PI * rho);
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = g[i * N + (threadIdx.y + m * BLOCK_SIZE)];
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dy * double(e + BLOCK_SIZE * m) * dky) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < N)
	{
		g[tid] = 1.0 + (y_val * c);
	}
}

__global__ void fft_cossine_x_integral(
	const double *__restrict__ g,
	double *__restrict__ S,
  int * index)
{
  int j = *index;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			x_shfl_src = g[(threadIdx.x + m * BLOCK_SIZE) * N + j] - 1.0;
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dx * double(e + BLOCK_SIZE * m) * dkx) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		S[tid] = y_val;
	}
}

__global__ void fft_cossine_y_integral(
	double *__restrict__ S,
  int * index)
{
  int i = *index;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
	double c = rho * dx * dy;
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = S[i * N + (threadIdx.y + m * BLOCK_SIZE)];
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//#pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dy * double(e + BLOCK_SIZE * m) * dky) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		S[tid] = 1.0 + c * y_val;
	}
}

__global__ void ifft_omega_x_integral(
	double *__restrict__ omega,
	const double *__restrict__ k2,
	const double *__restrict__ S,
  int * index)
{
  int j = *index;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;

  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			double aux = 1.0 - 1.0 / S[(threadIdx.x + m * BLOCK_SIZE) * N + j];
			x_shfl_src = -0.25 * k2[(threadIdx.x + m * BLOCK_SIZE) * N + j] * (2. * S[(threadIdx.x + m * BLOCK_SIZE) * N + j] + 1.0) * aux * aux;
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//#pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dx * double(e + BLOCK_SIZE * m) * dkx) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}

		__syncthreads();
	}

	if (tid < N)
	{
		omega[tid] = y_val;
	}
}

__global__ void ifft_omega_y_integral(
	double *__restrict__ omega,
  int * index)
{
  int i = *index;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
	double c = dkx * dky / (2.0 * M_PI * 2.0 * M_PI * rho);
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = omega[i * N + (threadIdx.y + m * BLOCK_SIZE)];
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dy * double(e + BLOCK_SIZE * m) * dky) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}
	if (tid < N)
	{
		omega[tid] = y_val * c;
	}
}

__global__ void fft_Vph_x_integral(
	const double *__restrict__ V,
	const double *__restrict__ second_term,
	const double *__restrict__ g,
	const double *__restrict__ omega,
	double *__restrict__ Vph,
  int * index)
{
  int j = *index;
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.x) < N)
		{
			double aux1 = g[(threadIdx.x + m * BLOCK_SIZE) * N + j] * V[(threadIdx.x + m * BLOCK_SIZE) * N + j];
			double aux2 = second_term[(threadIdx.x + m * BLOCK_SIZE) * N + j];
			double aux3 = (g[(threadIdx.x + m * BLOCK_SIZE) * N + j] - 1.0) * omega[(threadIdx.x + m * BLOCK_SIZE) * N + j];
			x_shfl_src = aux1 + aux2 + aux3;
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//        #pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dx * double(e + BLOCK_SIZE * m) * dkx) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		Vph[tid] = y_val;
	}
}

__global__ void fft_Vph_y_integral(
	double *__restrict__ Vph,
  int * index)
{
  int i = *index;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;
	double c = rho * dx * dy;
  #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = Vph[i * N + (threadIdx.y + m * BLOCK_SIZE)];
		}
		else
		{
			x_shfl_src = 0.0;
		}
		__syncthreads();

		//#pragma unroll
		for (int e = 0; e < 32; ++e)
		{
			// --- Column-major ordering - faster
			x_shfl_dest = __shfl_sync(0xffffffff, x_shfl_src, e);
			// y_val += d_j0table[tid * nCols + (e + BLOCK_SIZE * m)] * x_shfl_dest;
			y_val += cos(double(tid) * dy * double(e + BLOCK_SIZE * m) * dky) * x_shfl_dest;
			// --- Row-major ordering - slower
			// y_val += d_V_ph_k[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
		}
		__syncthreads();
	}

	if (tid < N)
	{
		Vph[tid] = c * y_val;
	}
}

__global__ void kernel_compute_second_term(double * g, double * second_term)
{
	double c1 = 0.25 / (60. * dx * 60. * dx);
	double c2 = 0.25 / (60. * dy * 60. * dy);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < N - 7; i += blockDim.x * gridDim.x)
	{
		for (int j = blockIdx.y * blockDim.y + threadIdx.y;  j < N - 7; j += blockDim.y * gridDim.y)
		{
			double grad_x = -147. * g[i * N + j] + 360. * g[(i + 1) * N + j] - 450. * g[(i + 2) * N + j] + 400 * g[(i + 3)* N + j];
			grad_x += -225. * g[(i + 4) * N + j] + 72. * g[(i + 5) * N + j] - 10. * g[(i + 6) * N + j];
			double grad_y = -147. * g[i * N + j] + 360. * g[i * N + j + 1] - 450. * g[i * N + j + 2] + 400 * g[i * N + j + 3];
			grad_y += -225. * g[i * N + j + 4] + 72. * g[i * N + j + 5] - 10. * g[i * N + j + 6];		   
			second_term[i * N + j] = (c1 * grad_x * grad_x + c2 * grad_y * grad_y) / (g[i * N + j]);
		}
	}
}

__global__ void update_S(double * S, const double *__restrict__ k2, const double *__restrict__ Vph)
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < N * N; i += blockDim.x * gridDim.x)
	{
		S[i] = (1.0 - dt) * S[i] + dt * sqrt(k2[i] / (k2[i] + 4.0 * Vph[i]));
	}
}

/*void FFT_g2S(const double * g, double * S, cudaStream_t * streams_x, cudaStream_t * streams_y, dim3 numBlocks, dim3 threadsPerBlock)
{
  #pragma unroll
  for (int i = 0; i < h_N; i++)
  {
    fft_cossine_x_integral<<<numBlocks, threadsPerBlock, 0, streams_x[i]>>>(g, S, i);
  }
  #pragma unroll
  for (int i = 0; i < h_N; i++) 
  {
    CUDA_CHECK(cudaStreamSynchronize(streams_x[i]));
  }  
  #pragma unroll
  for (int i = 0; i < h_N; i++)
  {
    fft_cossine_y_integral<<<numBlocks, threadsPerBlock, 0, streams_y[i]>>>(S, i);
  }  
  #pragma unroll
  for (int i = 0; i < h_N; i++) 
  {
    CUDA_CHECK(cudaStreamSynchronize(streams_y[i]));
  }
}*/

void IFFT_S2g(double * g, 
	const double * S, 
	cudaEvent_t * events_x, 
	cudaEvent_t * events_y, 
	cudaStream_t * streams_x, 
	cudaStream_t * streams_y, 
	dim3 numBlocks, 
	dim3 threadsPerBlock,
	int * index)
{
  for (int i = 0; i < h_N; i++)
  {
    ifft_cossine_x_integral<<<numBlocks, threadsPerBlock, 0, streams_x[i]>>>(g, S, &index[i]);
		cudaEventRecord(events_x[i], streams_x[i]);
  }
  for (int i = 0; i < h_N; i++) 
  {
    cudaStreamWaitEvent(streams_x[i], events_x[i]);
  }  
  for (int i = 0; i < h_N; i++)
  {
    ifft_cossine_y_integral<<<numBlocks, threadsPerBlock, 0, streams_y[i]>>>(g, &index[i]);
		cudaEventRecord(events_y[i], streams_y[i]);
  }  
  for (int i = 0; i < h_N; i++) 
  {
    CUDA_CHECK(cudaStreamSynchronize(streams_y[i]));
  }
}

void compute_second_term(double * g, double * second_term, dim3 numBlocks, dim3 threadsPerBlock)
{
	kernel_compute_second_term<<<numBlocks, threadsPerBlock>>>(g, second_term);
}

void compute_omega(double * omega, double * k2, double * g, double * S, cudaEvent_t * events_x, cudaEvent_t * events_y, cudaStream_t * streams_x, cudaStream_t * streams_y, dim3 numBlocks, dim3 threadsPerBlock, int * index)
{
  for (int i = 0; i < h_N; i++)
  {
    ifft_omega_x_integral<<<numBlocks, threadsPerBlock, 0, streams_x[i]>>>(omega, k2, S, &index[i]);
		cudaEventRecord(events_x[i], streams_x[i]);
  }
  for (int i = 0; i < h_N; i++) 
  {
    cudaStreamWaitEvent(streams_x[i], events_x[i]);
  }  
  for (int i = 0; i < h_N; i++)
  {
    ifft_omega_y_integral<<<numBlocks, threadsPerBlock, 0, streams_y[i]>>>(omega, &index[i]);
		cudaEventRecord(events_y[i], streams_y[i]);
  }  
  for (int i = 0; i < h_N; i++) 
  {
    cudaStreamWaitEvent(streams_y[i], events_y[i]);
  }	
}


void compute_Vph_k(	const double *__restrict__ V,
										const double *__restrict__ second_term,
										const double *__restrict__ g,
										const double *__restrict__ omega,
										double *__restrict__ Vph,
										cudaEvent_t * events_x, 
										cudaEvent_t * events_y, 
										cudaStream_t * streams_x, 
										cudaStream_t * streams_y, 
										dim3 numBlocks, 
										dim3 threadsPerBlock,
										int * index)
{
  for (int i = 0; i < h_N; i++)
  {
    fft_Vph_x_integral<<<numBlocks, threadsPerBlock, 0, streams_x[i]>>>(V, second_term, g, omega, Vph, &index[i]);
		cudaEventRecord(events_x[i], streams_x[i]);
  }
  for (int i = 0; i < h_N; i++) 
  {
    cudaStreamWaitEvent(streams_x[i], events_x[i]);
  }  
  for (int i = 0; i < h_N; i++)
  {
    fft_Vph_y_integral<<<numBlocks, threadsPerBlock, 0, streams_y[i]>>>(Vph, &index[i]);
		cudaEventRecord(events_y[i], streams_y[i]);
  }  
  for (int i = 0; i < h_N; i++) 
  {
    cudaStreamWaitEvent(streams_y[i], events_y[i]);
  }	
}

