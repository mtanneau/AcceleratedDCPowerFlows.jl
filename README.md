# AcceleratedDCPowerFlows.jl

This repository contains accelerated implementations of DC power flow computations.
The package currently supports:
* PTDF matrix computation, and PTDF-based flow computation
* LODF matrix computation, and LODF-based post-contingency flow computation
    Note that only single-branch outages that do not trigger loss of connectivity are supported.

Both PTDF/LODF support full matrix formation, as well as a sparsity-exploiting lazy implementation.
Lazy implementations provide substantial time and memory savings as they avoid forming any dense matrix.

## GPU acceleration

The package supports GPU acceleration on CUDA-compatible GPUs.
