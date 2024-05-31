__global__ void TEST_fft_cossine_x_integral(
	const double *__restrict__ g,
	double *__restrict__ S)
{  
	for (int j = blockIdx.y * blockDim.y + threadIdx.y;  j < N; j += blockDim.y * gridDim.y)
	{
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
}