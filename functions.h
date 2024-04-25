#ifndef FUNCTIONS
#define FUNCTIONS

extern int N;
extern int inv_N2;
extern double L;
extern double h;
extern double U;
extern double rho;
extern double dt;

void potential_U(double * x, double * y, double * V)
{
  #pragma omp parallel for
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < N; j++)
    {
      V[i * N + j] = U * exp(-pow(x[i * N + j], 2) - pow(y[i * N + j], 2));
    }
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