#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Initialize matrix with random values
void init_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

// CPU matrix multiplication for verification
void matrix_mult_cpu(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify results between CPU and GPU
bool verify_results(float *cpu_result, float *gpu_result, int size) {
    const float epsilon = 1e-3f;  // Larger epsilon due to floating point precision
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

// Print a small portion of the matrix for debugging
void print_matrix_sample(float *matrix, int rows, int cols, const char* name) {
    printf("%s (showing top-left 4x4):\n", name);
    for (int i = 0; i < min(4, rows); i++) {
        for (int j = 0; j < min(4, cols); j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

#endif // MATRIX_UTILS_H