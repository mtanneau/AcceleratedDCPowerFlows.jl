# AcceleratedDCPowerFlows.jl

[![codecov](https://codecov.io/gh/mtanneau/AcceleratedDCPowerFlows.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mtanneau/AcceleratedDCPowerFlows.jl)

This repository contains accelerated implementations of DC power flow computations.
The package currently supports:
* PTDF matrix computation, and PTDF-based flow computation
* LODF matrix computation, and LODF-based post-contingency flow computation
    Note that only single-branch outages that do not trigger loss of connectivity are supported.

Both PTDF/LODF support full matrix formation, as well as a sparsity-exploiting lazy implementation.
Lazy implementations provide substantial time and memory savings as they avoid forming any dense matrix.

## GPU acceleration

The package supports GPU acceleration.
At the moment, only CUDA devices are supported.

To perform an operation on the GPU, specify the backend as follows
```julia
network = ...  # build network data structure

A = branch_incidence_matrix(network)  # defaults to CPU

# Explicitly request CPU
using KernelAbstractions
A = branch_incidence_matrix(CPU(), network)

# Build on CUDA GPU (if available)
using CUDA, CUDSS  # we need both CUDA and CUDSS to be loaded
                   # in order to load CUDA extensions
A = branch_incidence_matrix(CUDA.CUDABackend(), network)
get_backend(A)     # CUDA.CUDABackend
```
