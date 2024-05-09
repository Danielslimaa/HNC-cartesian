#ifndef FUNCTIONS
#define FUNCTIONS

extern long int N;
extern double inv_N2;
extern double L;
extern double dx, dy, dkx, dky;
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
      x[i * N + j] = (-L / 2) + (i - 1) * dx;
      y[i * N + j] = (-L / 2) + (j - 1) * dy;
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

void compute_omega(fftw_plan omega_to_omega, double * k2, double * S, double * omega)
{
  double c = dkx * dky * ( 1.0 / (2. * M_PI * 2.0 * M_PI * rho) ) * inv_N2;
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    omega[i] = - c * 0.25 * k2[i] * ( 2. * S[i] + 1. ) * ( 1. - (1. / S[i]) ) * ( 1. - (1. / S[i]) ); 
  }  
  fftw_execute(omega_to_omega);
}

double compute_error(double * new_g, double * g)
{
  double sum, tmp= 0;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for (int i = 0; i < N * N; i++)
  {
    tmp = abs(new_g[i] - g[i]); 
    sum += tmp; 
  } 
  return sum * dx * dy / dt;
}

void laplace(double * g, double * Lg)
{
  double inv_hh = 1. / (dx * dx); 
  #pragma omp parallel for 
  for (int i = 0; i < N / 2; i++)
  {
    for (int j = 0; j < N / 2; j++)
    {
      Lg[i * N + j] = g[i * N + j + 1] - g[i * N + j] + g[i * N + j - 1]; // del_del_x
      Lg[i * N + j] += g[(i + 1) * N + j] - g[i * N + j] + g[(i - 1) * N + j]; // del_del_y
      Lg[i * N + j] *= inv_hh;
      Lg[(N - 1 - i) * N + j] = Lg[i * N + j];
      Lg[i * N + (N - 1 - j)] = Lg[i * N + j];
      Lg[(N - 1 - i) * N + (N - 1 - j)] = Lg[i * N + j];
    } 
  } 
}

void update_g(double * Lg, double * V, double * omega, double * g)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    g[i] = (1. - dt) * g[i] + dt * (-Lg[i] + V[i] * g[i] + omega[i] * g[i]); //Eq.(7) 
  }
  return;
}

void compute_S(fftw_plan g_to_S, double * g, double * new_S)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    new_S[i] = g[i] * g[i] - 1.0; //Because g[] = sqrt{g[]}
  }
  fftw_execute(g_to_S); //Actually it is a FFTW_REDFT00 made inplace with new_S -> new_S
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    new_S[i] = 1.0 + rho * new_S[i] * dkx * dky * inv_N2;
  }
  return;  
}

void initialize_g_S(double * g, double * S)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    S[i] = 1.0;
    g[i] = 1.10;
  }
  return;  
}


#endif