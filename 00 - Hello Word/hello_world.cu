#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that prints from each thread
__global__ void hello_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main() {
    printf("=== CUDA Hello World ===\n");
    
    // Launch configuration
    int numBlocks = 2;
    int threadsPerBlock = 4;
    
    printf("Launching %d blocks with %d threads each...\n", numBlocks, threadsPerBlock);
    
    // Launch kernel
    hello_kernel<<<numBlocks, threadsPerBlock>>>();
    
    // Wait for GPU to finish and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("GPU execution completed successfully!\n");
    
    return 0;
}
