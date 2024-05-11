#ifndef FUNCTIONS
#define FUNCTIONS

extern long int N;
extern double inv_N2;
extern double L;
extern double dx, dy, dkx, dky;
extern double U;
extern double rho;
extern double dt;

void printer_field(double * x, double * y, double *vetor, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N * N; ++i)
	{
		myfile << x[i] << " " << y[i] << " " << vetor[i] << "\n";
	}
	myfile.close();
	return;
}

void printer_field_transversal_view(double * x, double * y, double *vetor, const char *name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = N / 2; i < N; ++i)
	{
		myfile << vetor[i * N + N / 2 - 1] << "\n";
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
      // Option 1: [0, L - h] x [0, L - h]
      x[i * N + j] = 0 + i * dx; 
      y[i * N + j] = 0 + j * dy;
      
      // Option 2, the traditional one: [-L / 2, L / 2] x [-L / 2, L / 2]
      //x[i * N + j] = (-L / 2) + (i - 1) * dx; 
      //y[i * N + j] = (-L / 2) + (j - 1) * dy;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      kx[i * N + j] = i * 2.0 * M_PI / (2. * L);
    }    
    //for (int j = N / 2; j < N; j++)
    //{
    //  kx[i * N + j] = (i - N) * 2 * M_PI / L;
    //}
  }
  
  #pragma omp parallel for
  for (int j = 0; j < N; j++)
  {    
    for (int i = 0; i < N; i++)
    {
      ky[i * N + j] = j * 2.0 * M_PI / (2. * L);
    }    
    //for (int i = N / 2; i < N; i++)
    //{
    //  ky[i * N + j] = (j - N) * 2.0 * M_PI / L;
    //}
  }

  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {    
    k2[i] = kx[i] * kx[i] + ky[i] * ky[i];
  }  
}

void potential_V(double * x, double * y, double * V)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    V[i] = U * exp(-x[i] * x[i] - y[i] * y[i]);
  }
}

void compute_omega(fftw_plan omega_to_omega, double * k2, double * S, double * omega)
{
  double c = dkx * dky * ( 1.0 / (2. * M_PI * 2.0 * M_PI * rho) ) * 0.25;
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    omega[i] = - c * k2[i] * ( 2. * S[i] + 1. ) * ( 1. - (1. / S[i]) ) * ( 1. - (1. / S[i]) ); 
  }  
  fftw_execute(omega_to_omega);
}

void laplace_finite_difference(double * g, double * Lg)
{
  double inv_hh = 1. / (dx * dx); 
  #pragma omp parallel for 
  for (int i = 0; i < N - 2; i++)
  {
    for (int j = 0; j < N - 2; j++)
    {
      Lg[i * N + j] = g[i * N + j + 2] - 2. * g[i * N + j + 1] + g[i * N + j]; // del_del_y
      Lg[i * N + j] += g[(i + 2) * N + j] - 2. * g[(i + 1) * N + j] + g[(i) * N + j]; // del_del_x
      Lg[i * N + j] *= inv_hh;
      //Lg[i * N + N / 2 + j - 1] = Lg[i * N + j];
      //Lg[(N / 2 + i - 1) * N + N / 2 + j - 1] = Lg[i * N + j];
      //Lg[(N / 2 + i - 1) * N + j] = Lg[i * N + j];
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
    new_S[i] = g[i] * g[i] - 1.0; //Because g[] is actually being  sqrt{g} here
  }
  fftw_execute(g_to_S); //Actually it is a FFTW_REDFT00 made inplace with new_S -> new_S
  double c = rho * dx * dy;
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    new_S[i] = 1.0 + c * new_S[i];
  }
  return;  
}

void initialize_g_S(double * g, double * S)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    S[i] = 1.0;
    g[i] = 1.0;
  }
  return;  
}

double compute_g_error(double * new_g, double * g)
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

double compute_S_error(double * new_S, double * S)
{
  double sum, tmp= 0;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for (int i = 0; i < N * N; i++)
  {
    tmp = abs(new_S[i] - S[i]); 
    sum += tmp; 
  } 
  return sum * dkx * dky / dt;
}

#endif