#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// CUDA kernel for vector addition
__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// CPU version for comparison
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Verify results
bool verify_results(float *cpu_result, float *gpu_result, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1000000;  // 1 million elements
    const int bytes = N * sizeof(float);
    
    printf("=== Vector Addition: %d elements ===\n", N);
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_cpu = (float*)malloc(bytes);
    float *h_c_gpu = (float*)malloc(bytes);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Grid size: %d blocks, Block size: %d threads\n", gridSize, blockSize);
    
    // GPU computation with timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // CPU computation with timing
    clock_t cpu_start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Verify results
    bool correct = verify_results(h_c_cpu, h_c_gpu, N);
    
    // Print results
    printf("\nResults:\n");
    printf("CPU time: %.2f ms\n", cpu_time);
    printf("GPU time: %.2f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_a); free(h_b); free(h_c_cpu); free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
