# 00 - Hello World

## Overview
This is your first CUDA program! This project introduces the absolute basics of GPU programming with CUDA.

## Learning Objectives
- Understand basic CUDA concepts
- Learn how to write a simple kernel function
- Understand memory allocation on GPU
- Learn basic error checking

## What You'll Build
A simple "Hello World" program that runs on the GPU and demonstrates:
- Basic kernel launch syntax
- Thread identification
- Simple output from GPU threads

## Files
- `hello_world.cu` - Main CUDA program
- `Makefile` - Build configuration

## Concepts Covered
- CUDA kernels (`__global__` functions)
- Thread indexing (`threadIdx`, `blockIdx`)
- Basic memory management (`cudaMalloc`, `cudaFree`)
- Kernel launch configuration (`<<<blocks, threads>>>`)

## Prerequisites
- Basic C/C++ knowledge
- CUDA toolkit installed
- NVIDIA GPU with compute capability 3.0+

## Next Steps
After completing this project, move on to "01 - Vector Addition" to learn about parallel data processing.