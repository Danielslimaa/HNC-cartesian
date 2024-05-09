#ifndef FUNCTIONS
#define FUNCTIONS

extern long int N;
extern double inv_N2;
extern double L;
extern double h;
extern double U;
extern double rho;
extern double dt;

void printer_vector(double * x, double * y, double *vetor, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N * N; ++i)
	{
		myfile << x[i] << " " << y[i] << " " << vetor[i] << "\n";
        //std::cout << h_x[i] << "," << h_y[i] << "," << h_vetor[i] << std::endl;
	}
	myfile.close();
	return;
}

void geometry(double * x, double * y, double * kx, double * ky, double * k2)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      x[i * N + j] = (-L / 2) + (i - 1) * h;
      y[i * N + j] = (-L / 2) + (j - 1) * h;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N / 2; j++)
    {
      kx[i * N + j] = i * 2 * M_PI / L;
    }    
    for (int j = N / 2; j < N; j++)
    {
      kx[i * N + j] = (i - N) * 2 * M_PI / L;
    }
  }
  
  #pragma omp parallel for
  for (int j = 0; j < N; j++)
  {    
    for (int i = 0; i < N / 2; i++)
    {
      ky[i * N + j] = j * 2.0 * M_PI / L;
    }    
    for (int i = N / 2; i < N; i++)
    {
      ky[i * N + j] = (j - N) * 2.0 * M_PI / L;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {    
    k2[i] = kx[i] * kx[i] + ky[i] * ky[i];
  }  
}

void potential_U(double * x, double * y, double * V)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    V[i] = U * exp(-x[i] * x[i] - y[i] * y[i]);
  }
}

double compute_error(fftw_complex * new_g, fftw_complex * g)
{
  double sum, tmp= 0;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for (int i = 0; i < N * N; i++)
  {
    tmp = abs(new_g[i][0] - g[i][0]); 
    sum += tmp; 
  } 
  return sum * h * h / dt;
}

#endif