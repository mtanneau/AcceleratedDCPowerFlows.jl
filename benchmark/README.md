# Benchmark Suite

This directory contains benchmark drivers for `AcceleratedDCPowerFlows.jl`.

## Setup

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

## Available Benchmarks

- `ptdf`: see `ptdf/README.md`
- `lodf`: see `lodf/README.md`
