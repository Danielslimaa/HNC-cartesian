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
#include "functions.h"

//g++ K.cpp -w -lfftw3_omp -lfftw3 -fopenmp -lm -O3

int main(void){
  N = 1 << 10;
  read_field(g, "g_field_only.dat");

}