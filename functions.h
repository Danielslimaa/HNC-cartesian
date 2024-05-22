#ifndef FUNCTIONS
#define FUNCTIONS
#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <fftw3.h>

int N;
double inv_N2;
double L; 
double dx, dy, dkx, dky;
double U;
double rho;
double dt;
double energy, new_energy;
bool condition;
double tolerance;
double lambda, new_lambda;

void printer_field(double * x, double * y, double *vetor, const char * name)
{
	std::ofstream myfile(name);
	for (int i = 0; i < N * N; ++i)
	{
		myfile << x[i] << '\t' << y[i] << '\t' << vetor[i] << '\n';
	}
	myfile.close();
	return;
}

void printer_field_transversal_view(double * x, double * y, double *vetor, const char * name)
{
	std::ofstream myfile;
	myfile.open(name);
	for (int i = 0; i < N; ++i)
	{
		myfile << vetor[i * N + 0] << "\n";
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

void potential_V(double * x, double * y, double * k2, double * V, const char * name)
{
  if ( name == "GEM2")
  {
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      V[i] = U * exp(-x[i] * x[i] - y[i] * y[i]);
    }
  }
  if ( name == "Rydberg")
  {
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      double r = sqrt(x[i] * x[i] + y[i] * y[i]);
      V[i] = U * (1. / (1. + pow(r, 6)));
    }
  }  
  if (name == "Dipolar_Zillinger")
  {
    double C_h = 0.33;
    double theta = 1.08;
    double sin_theta = sin(theta);
    double dl = 1;//L / double(N);
    printf("dl = %1.6f\n", dl);

    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      double r = sqrt(x[i] * x[i] + y[i] * y[i] + dl * dl);      
      double inverse_r = pow(r, -1);
      double inverse_r3 = pow(r, -3);
      double inverse_r5 = pow(r, -5);
      double x_coordinate = x[i];
      V[i] = ( inverse_r3 - (3. * x_coordinate * sin_theta * x_coordinate * sin_theta * inverse_r5) ) + pow(C_h * inverse_r, 12);
      
    }
  } 
  if (name == "QC_hexagonal")
  {
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      double k = sqrt(k2[i]);
      if(k <= 0.749)
      {
        V[i] = 5.0;
      }      
      if(k > 0.749 && k <= 1.251)
      {
        V[i] = -1.0 + 96.0 * pow((-1.0 + k), 2);
      }
      if(k > 1.251 && k <= 1.749)
      {
        V[i] = 5.0;
      }
      if (k > 1.749 && k <= 2.17908)
      {
        V[i] = -1.0 + 181.433 * (-sqrt(2 + sqrt(3)) + k) * (-sqrt(2 + sqrt(3)) + k);
      }
      if(k > 2.17908)
      {
        V[i] = 10.0;
      }
    }
    printer_field_transversal_view(x, y, V, "fftV.dat");
    fftw_plan p = fftw_plan_r2r_2d(N, N, V, V, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);
    fftw_destroy_plan(p);
      #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      V[i] *= dkx * dky;
    }
    printer_field_transversal_view(x, y, V, "V.dat");

  }  
  if (name == "QC_dodecagonal")
  {
    #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      double k = sqrt(k2[i]);
      if(k <= 0.749)
      {
        V[i] = 5.0;
      }      
      if(k > 0.749 && k <= 1.251)
      {
        V[i] = -1.0 + 96.0 * pow((-1.0 + k), 2);
      }
      if(k > 1.251 && k <= 1.749)
      {
        V[i] = 5.0;
      }
      if (k > 1.749 && k <= 2.167)
      {
        V[i] = -1.0 + 229.815 * (-sqrt(2 + sqrt(3)) + k) * (-sqrt(2 + sqrt(3)) + k);
      }
      if(k > 2.167)
      {
        V[i] = 10.0;
      }
    }
    fftw_plan p = fftw_plan_r2r_2d(N, N, V, V, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);
    fftw_destroy_plan(p);
      #pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
      V[i] *= dkx * dky;
    }
  }  
}

void compute_omega(fftw_plan omega_to_omega, double * k2, double * S, double * omega)
{
  double c = - dkx * dky * ( 1.0 / (2. * M_PI * 2.0 * M_PI * rho) ) * 0.25;
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    double aux = ( 1. - (1. / S[i]) );
    omega[i] = c * k2[i] * ( 2. * S[i] + 1. ) * aux * aux; 
  }  
  fftw_execute(omega_to_omega);
}

void initialize_g_S(double * x, double * y, double * g, double * S)
{
  #pragma omp parallel for
  for (int i = 0; i < N * N; i++)
  {
    S[i] = 1.0;// - exp(-x[i] * x[i] - y[i] * y[i]);
    g[i] = S[i];
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

void compute_Vph(double * V, double * g, double * omega, double * Vph)
{
  #pragma omp parallel for 
  for (int i = 0; i < N - 4; i++)
  {
    for (int j = 0; j < N - 4; j++)
    {
      Vph[i * N + j] = g[i * N + j] * V[i * N + j];
      double dely_g = 2. * g[i * N + j + 3] - 9. * g[i * N + j + 2] + 18. * g[i * N + j + 1] - 11. * g[i * N + j];
      double delx_g = 2. * g[(i + 3) * N + j] - 9. * g[(i + 2) * N + j] + 18. * g[(i + 1) * N + j] - 11. * g[i * N + j];
      Vph[i * N + j] += (delx_g * delx_g + dely_g * dely_g) / (6. * dx * 6. * dx * 4. * g[i * N + j]);
      Vph[i * N + j] += (  g[i * N + j] - 1.  ) * omega[i * N + j];
    } 
  }   
  return;
}

void update_S(fftw_plan Vph_to_Vph, double * k2, double * Vph, double * S)
{
  double c = rho * dx * dy;
  fftw_execute(Vph_to_Vph);
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    Vph[i] *= c;
    double dS = sqrt( k2[i] / ( k2[i] + 4. * Vph[i] ) );
    S[i] = (1. - dt) * S[i] + dt * dS; 
  } 
  return;
}

void compute_g(fftw_plan g_to_g, double * S, double * g)
{
  double c = dkx * dky / (2. * M_PI * 2. * M_PI * rho);
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    g[i] = S[i] - 1.; 
  }
  fftw_execute(g_to_g);
  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    g[i] = 1. + c * g[i];
  } 
  return;
}

double compute_energy(double * k2, double * g, double * S, double * V)
{
  double c1 = (rho / 2.) * dx * dy;
  double c2 = - (1. / 8.) * ( 1. / (2. * M_PI * 2. * M_PI * rho) ) * dkx * dky;
  double c3 = - (rho / 2.) * 0.25;
  double sum, tmp = 0;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for (int i = 0; i < N - 2; i++)
  {
    for (int j = 0; j < N - 2; j++)
    {
      double aux = S[i * N + j] - 1.;
      tmp = c1 * V[i * N + j] * g[i * N + j] + c2 * k2[i * N + j] * aux * aux * aux / S[i * N + j];
      tmp += c3 * g[i * N + j] * ( log(g[i * N + j + 2]) - 2. * log(g[i * N + j + 1]) + log(g[i * N + j]) ); // delyy_g
      tmp += c3 * g[i * N + j] * ( log(g[(i + 2) * N + j]) - 2. * log(g[(i + 1) * N + j]) + log(g[(i) * N + j]) ); // delxx_g

      sum += tmp;
    } 
  } 
  return sum;
}

void compute_W_part(double * fft_sqrt_g, 
                    double * W, 
                    double * f,
                    fftw_plan W_to_W)
{
  double c1 = dx * dy;
  double c2 = dkx * dky / (2. * M_PI * 2. * M_PI);
 
  #pragma for omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    W[i] = c1 * W[i] * f[i];
  }
  fftw_execute(W_to_W);
  #pragma for omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    W[i] = c2 * fft_sqrt_g[i] * W[i];
  }  
  fftw_execute(W_to_W);
}

void compute_kinetic1(double * expAx, 
                    double * expAy, 
                    double * f, 
                    fftw_plan p_x, 
                    fftw_plan p_y)
{
  fftw_execute(p_x);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAx[i];
  }  
  fftw_execute(p_x);
  fftw_execute(p_y);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAy[i];
  }  
  fftw_execute(p_y);
}

void compute_kinetic2(double * expAx, 
                    double * expAy, 
                    double * f, 
                    fftw_plan p_x, 
                    fftw_plan p_y)
{
  fftw_execute(p_y);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAy[i];
  }  
  fftw_execute(p_y);
  fftw_execute(p_x);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAx[i];
  }  
  fftw_execute(p_x);
}

void compute_part1(double * expAx, 
                    double * expAy, 
                    double * f,
                    double * sqrt_g, 
                    fftw_plan p_x, 
                    fftw_plan p_y)
{
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= 1;//sqrt_g[i];
  }
  fftw_execute(p_x);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAx[i];
  }  
  fftw_execute(p_x);
  fftw_execute(p_y);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] *= expAy[i];
  }  
  fftw_execute(p_y);
}

void normalize_f(double * f)
{
  double sum, tmp = 0;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for(int i = 0; i < N * N; i++)
  {
    tmp = f[i] * f[i];
    sum += tmp;
  }
  sum *= dx * dy;
  sum = sqrt(sum);
  #pragma omp parallel for
  for(int i = 0; i < N * N; i++)
  {
    f[i] /= sum;
  }
}

void printer_loop_f(long int counter, double * k2, double * g, double * f, double * V, double * omega, double * W, fftw_plan W_to_W, fftw_plan f_to_f)
{

  if(counter%1000 == 0)
  {
    double sum, tmp = 0;
    #pragma omp parallel for reduction(+ : sum) private(tmp)
    for(int i = 0; i < N * N; i++)
    {
      tmp = f[i] * f[i];
      sum += tmp;
    }
    double ff = sum * dx * dy;
    double * Lf = new double[N * N];
    #pragma omp parallel for
    for(int i = 0; i < N * N; i++)
    {
      Lf[i] = (V[i] + omega[i] + W[i]) * f[i];
    }
    fftw_execute(f_to_f);
    #pragma omp parallel for
    for(int i = 0; i < N * N; i++)
    {
      f[i] *= -k2[i] * inv_N2;
    }
    fftw_execute(f_to_f);
    #pragma omp parallel for
    for(int i = 0; i < N * N; i++)
    {
      Lf[i] += f[i];
    }  
    sum = 0;
    tmp = 0;
    #pragma omp parallel for reduction(+ : sum) private(tmp)
    for(int i = 0; i < N * N; i++)
    {
      tmp = f[i] * Lf[i];
      sum += tmp;
    }   
    double fLf = sum * dx * dy;
    new_lambda = fLf / ff;
    double error = abs(new_lambda - lambda) / dt;
    printf("counter = %ld, lambda = %1.4f, error = %1.6e, fLf = %1.6f, ff = %1.6f \n", counter, new_lambda, error, fLf, ff);
    lambda = new_lambda;
    condition = error > tolerance;
    delete[] Lf;
  }
}


void print_loop(double * x, double * y, double * k2, double * g, double * V, double * S, double * new_S, long int counter)
{
  if(counter%1 == 0 or counter == 1)
  {
    new_energy = compute_energy(k2, g, new_S, V);
    double error = abs(new_energy - energy) / dt;
    bool aux_condition = error > tolerance;
    printf("t = %ld, de/dt = %1.4e, e = %1.6f\n", counter, error, new_energy);
    if(counter > 1)
    {
      printer_field_transversal_view(x, y, S, "S.dat");
      printer_field_transversal_view(x, y, g, "g.dat");
      //printer_field(x, y, g, "g.dat");
      //printer_field(x, y, S, "S.dat");
    }
    memcpy(new_S, S, N * N * sizeof(double));
    energy = new_energy;
    condition = aux_condition;
  }
}

void read_field(double * x, double *y, double * vetor, const char * name)
{
	std::ifstream myfile(name);
  // Check if the file is open
  if (!myfile.is_open()) {
      std::cerr << "Error: Could not open the file." << std::endl;
  }
  else
  {
    for (int i = 0; i < N * N; ++i)
    {
      myfile >> x[i] >> y[i] >> vetor[i];
      myfile.ignore(1, '\t');
    }
  }
	myfile.close();
}

void preliminaries(double * k2,
                double * g, 
                double * S, 
                double * omega, 
                double * W,
                double * Wx,
                double * Wy, 
                double * sqrt_g,
                double * fft_sqrt_g)
{
  double c1 = dkx * dky * ( 1.0 / (2. * M_PI * 2.0 * M_PI) );
  double c2 = dx * dy * rho;

  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    double aux1 = ( 1. - (1. / (S[i] * S[i] * S[i])));
    double aux2 = ( 1. - (1. / (S[i])));
    omega[i] = -c1 * 0.25 * k2[i] * ( 2. * S[i] + 1. ) * aux2 * aux2; 
    sqrt_g[i] = sqrt(g[i]);
    W[i] = -c1 * k2[i] * aux1;
  }  

  fftw_plan omega_to_omega = fftw_plan_r2r_2d(N, N, omega, omega, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);
  fftw_plan W_to_W = fftw_plan_r2r_2d(N, N, W, W, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);
  fftw_plan sqrt_g_to_sqrt_g = fftw_plan_r2r_2d(N, N, sqrt_g, fft_sqrt_g, FFTW_REDFT00, FFTW_REDFT00, FFTW_MEASURE);

  fftw_execute(W_to_W); // W(r - r')

  #pragma omp parallel for 
  for (int i = 0; i < N * N; i++)
  {
    W[i] = sqrt_g[i] * W[i]; // sqrt(g(r)) x W(r - r')
  }
  fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};
  int howmany = N;
  fftw_plan p_Wx = fftw_plan_many_r2r(1, &N, howmany, W, NULL, howmany, 1, Wx, NULL, howmany, 1, kinds, FFTW_MEASURE);
  fftw_plan p_Wy = fftw_plan_many_r2r(1, &N, howmany, W, NULL, 1, howmany, Wy, NULL, 1, howmany, kinds, FFTW_MEASURE);  

  fftw_execute(p_Wx);
  fftw_execute(p_Wy);
  fftw_execute(omega_to_omega); // omega(r')
  fftw_execute(sqrt_g_to_sqrt_g); // FFT{sqrt(g)}

  fftw_destroy_plan(omega_to_omega);
  fftw_destroy_plan(W_to_W);
  fftw_destroy_plan(sqrt_g_to_sqrt_g);
  fftw_destroy_plan(p_Wx);
  fftw_destroy_plan(p_Wy);
}

void S_from_g(double * g, double * S)
{
  double sum, tmp = 0;
  double c1 = dkx * dx;
  double c2 = dky * dy;
  double c3 = dx * dy * rho;
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for(int j = 0; j < N; j++)
  {
    sum = 0;
    for(int i = 0; i < N; i++)
    {
      tmp = (g[i * N + j] - 1) * cos(i * j * c1);
      sum += tmp;
    }
    S[i * N + j] = c3;
  }
  #pragma omp parallel for reduction(+ : sum) private(tmp)
  for(int i = 0; i < N; i++)
  {
    sum = 0;
    for(int j = 0; j < N; j++)
    {
      tmp = S[i * N + j] * cos(i * j * c2);
      sum += tmp;
    }
    S[i * N + j] = 1 + sum * c3;
  }  
}

#endif