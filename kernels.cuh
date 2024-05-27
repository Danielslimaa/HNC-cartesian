#pragma once
#define CUDA_GRID_STRIDE_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;  i < n; i += blockDim.x * gridDim.x)

#define MAX_BLKSZ 1024
#define WARPSZ 32
#define BLOCK_SIZE 32

#define PI M_PI

__constant__ int N;
__constant__ double h;
__constant__ double L;
__constant__ double rho;

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

__global__ void fft_cossine_x_integral(
	double *__restrict__ g,
	const double *__restrict__ S,
	double *sum_diff_g
    int * d_j)
{
    int j = *d_j;
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
		g[tid] = y_val;
	}
}

__global__ void ifft_cossine_y_integral(
	double *__restrict__ g,
	const double *__restrict__ S,
	double *sum_diff_g
    int * d_i)
{
    int i = *d_i;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;

    #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = S[i * N + (threadIdy.y + m * BLOCK_SIZE)] - 1.0;
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
		g[tid] = 1.0 + (y_val * dk * dk / (2.0 * M_PI * rho));
	}
}

__global__ void fft_cossine_y_integral(
	double *__restrict__ g,
	const double *__restrict__ S,
	double *sum_diff_g
    int * d_i)
{
    int i = *d_i;
	const unsigned int tid = threadIdx.y + blockIdx.y * blockDim.y;
	double x_shfl_src, x_shfl_dest;
	double y_val = 0.0;

    #pragma unroll
	for (unsigned int m = 0; m < ((N + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
	{
		if ((m * BLOCK_SIZE + threadIdx.y) < N)
		{
			x_shfl_src = g[i * N + (threadIdy.y + m * BLOCK_SIZE)] - 1.0;
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
		S[tid] = 1.0 + rho * dx * dy * y_val;
	}
}