# 01 - Vector Addition

## Overview
Learn parallel computing fundamentals by implementing vector addition on the GPU. This project introduces data parallelism and memory management.

## Learning Objectives
- Understand parallel data processing
- Learn host-device memory transfers
- Practice kernel programming with real data
- Understand performance considerations

## What You'll Build
A program that adds two large vectors in parallel on the GPU, demonstrating:
- Memory allocation on both host and device
- Data transfer between CPU and GPU
- Parallel computation across array elements
- Performance comparison with CPU implementation

## Files
- `vector_add.cu` - Main CUDA program with vector addition
- `Makefile` - Build configuration

## Concepts Covered
- `cudaMalloc()` and `cudaFree()` for device memory
- `cudaMemcpy()` for host-device data transfer
- Global memory access patterns
- Thread-to-data mapping
- Basic performance timing

## Key CUDA Concepts
- **Grid and Block dimensions**: How to map threads to data elements
- **Memory management**: Allocating and copying data efficiently
- **Kernel optimization**: Writing efficient parallel algorithms

## Prerequisites
- Completed "00 - Hello World"
- Understanding of arrays and loops in C/C++

## Next Steps
After mastering vector addition, proceed to "02 - Matrix Multiplication" to work with 2D data structures.