#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include "matrix_utils.h"

#define TILE_SIZE 16

// Naive matrix multiplication kernel
__global__ void matrix_mult_naive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized tiled matrix multiplication kernel using shared memory
__global__ void matrix_mult_tiled(float *A, float *B, float *C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * TILE_SIZE + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to make sure tiles are loaded
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_matrix_multiplication(int M, int N, int K) {
    printf("\n=== Matrix Multiplication: %dx%d * %dx%d ===\n", M, K, K, N);
    
    // Calculate sizes
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_naive = (float*)malloc(size_C);
    float *h_C_tiled = (float*)malloc(size_C);
    
    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // CPU computation
    printf("Running CPU implementation...\n");
    clock_t cpu_start = clock();
    matrix_mult_cpu(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    // GPU Naive implementation
    printf("Running GPU naive implementation...\n");
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mult_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naive_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // GPU Tiled implementation
    printf("Running GPU tiled implementation...\n");
    dim3 tiledBlockSize(TILE_SIZE, TILE_SIZE);
    dim3 tiledGridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    CUDA_CHECK(cudaEventRecord(start));
    matrix_mult_tiled<<<tiledGridSize, tiledBlockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float tiled_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_time, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));
    
    // Verify results
    bool naive_correct = verify_results(h_C_cpu, h_C_naive, M * N);
    bool tiled_correct = verify_results(h_C_cpu, h_C_tiled, M * N);
    
    // Print results
    printf("\nPerformance Results:\n");
    printf("CPU time:          %.2f ms\n", cpu_time);
    printf("GPU naive time:    %.2f ms (%.2fx speedup)\n", naive_time, cpu_time / naive_time);
    printf("GPU tiled time:    %.2f ms (%.2fx speedup)\n", tiled_time, cpu_time / tiled_time);
    printf("Tiled vs Naive:    %.2fx improvement\n", naive_time / tiled_time);
    printf("\nVerification:\n");
    printf("Naive GPU:         %s\n", naive_correct ? "PASSED" : "FAILED");
    printf("Tiled GPU:         %s\n", tiled_correct ? "PASSED" : "FAILED");
    
    // Print sample results
    print_matrix_sample(h_C_cpu, M, N, "CPU Result");
    print_matrix_sample(h_C_tiled, M, N, "GPU Tiled Result");
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_naive); free(h_C_tiled);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=== CUDA Matrix Multiplication Comparison ===\n");
    
    // Test with different matrix sizes
    run_matrix_multiplication(512, 512, 512);
    run_matrix_multiplication(1024, 1024, 1024);
    
    return 0;
}
