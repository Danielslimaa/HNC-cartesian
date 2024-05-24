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

int main() {
    int N = 4; // Original size
    int P = 2 * N; // Padded size (double the original size)

    // Allocate memory for the original and padded 2D arrays
    std::vector<std::vector<double>> original_data(N, std::vector<double>(N));
    std::vector<std::vector<double>> padded_data(P, std::vector<double>(P, 0.0));
    double* fft_result = (double*) fftw_malloc(sizeof(double) * N * N);
    double* fft_result_padded = (double*) fftw_malloc(sizeof(double) * P * P);

    // Fill the original data with some values (example)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            original_data[i][j] = sin(i) * cos(j);
        }
    }

    // Copy the original data into the top-left corner of the padded array
    for (int i = 0; i < N; ++i) {
        std::copy(original_data[i].begin(), original_data[i].end(), padded_data[i].begin());
    }

    // Create FFTW plans for 2D REDFT00 transformation (DCT-II)
    fftw_plan plan = fftw_plan_r2r_2d(N, N, &original_data[0][0], fft_result, FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_plan plan_padded = fftw_plan_r2r_2d(P, P, &padded_data[0][0], fft_result_padded, FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);

    // Execute the plans
    fftw_execute(plan);
    fftw_execute(plan_padded);

    // Normalize the FFT results
    double normalization_factor = N * N;
    double padded_normalization_factor = P * P;

    std::vector<std::vector<double>> fft_result_mag(N, std::vector<double>(N));
    std::vector<std::vector<double>> fft_result_padded_mag(P, std::vector<double>(P));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fft_result[i * N + j] /= normalization_factor;
            fft_result_mag[i][j] = fft_result[i * N + j];
        }
    }

    for (int i = 0; i < P; ++i) {
        for (int j = 0; j < P; ++j) {
            fft_result_padded[i * P + j] /= padded_normalization_factor;
            fft_result_padded_mag[i][j] = fft_result_padded[i * P + j];
        }
    }

    // Print the results
    std::cout << "Original Data:" << std::endl;
    for (const auto& row : original_data) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nDCT-II without Zero Padding (Normalized):" << std::endl;
    for (const auto& row : fft_result_mag) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nZero-Padded Data:" << std::endl;
    for (const auto& row : padded_data) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nDCT-II with Zero Padding (Normalized):" << std::endl;
    for (const auto& row : fft_result_padded_mag) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan_padded);
    fftw_free(fft_result);
    fftw_free(fft_result_padded);

    return 0;
}
