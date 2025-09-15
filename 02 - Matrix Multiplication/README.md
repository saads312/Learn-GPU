# 02 - Matrix Multiplication

## Overview
Dive into 2D parallel computing with matrix multiplication. This project introduces more complex memory access patterns and threading concepts.

## Learning Objectives
- Understand 2D thread indexing and grid layouts
- Learn about memory coalescing and access patterns
- Practice with shared memory for optimization
- Understand the importance of tile-based algorithms

## What You'll Build
An optimized matrix multiplication implementation featuring:
- Naive global memory implementation
- Tiled implementation using shared memory
- Performance comparison between approaches
- Different block size optimizations

## Files
- `matrix_mult.cu` - Matrix multiplication implementations
- `matrix_utils.h` - Helper functions for matrix operations
- `Makefile` - Build configuration

## Concepts Covered
- **2D thread indexing**: Using `threadIdx.x/y` and `blockIdx.x/y`
- **Shared memory**: `__shared__` keyword and synchronization
- **Memory coalescing**: Optimizing global memory access patterns
- **Tiling algorithms**: Breaking large problems into smaller chunks
- **Thread synchronization**: `__syncthreads()`

## Key CUDA Concepts
- **Block dimensions**: Working with 2D thread blocks
- **Shared memory optimization**: Reducing global memory traffic
- **Memory hierarchy**: Understanding different memory types
- **Occupancy**: Maximizing GPU utilization

## Prerequisites
- Completed "01 - Vector Addition"
- Understanding of matrix operations
- Basic knowledge of memory hierarchies

## Next Steps
Progress to "03 - Image Processing" to apply these concepts to real-world image manipulation tasks.