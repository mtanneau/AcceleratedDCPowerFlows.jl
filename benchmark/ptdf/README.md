# PTDF Benchmark

This directory contains the PTDF benchmark driver for `AcceleratedDCPowerFlows.jl`.

## Run

To execute this benchmark, run
```bash
julia --project=benchmark --threads <num_threads> benchmark/ptdf/benchmark_ptdf.jl
```
where `<num_threads>` is the number of CPU threads.

The script uses fixed constants near the top of the file for cases, backends,
solvers, benchmark durations, and output path.

## Export Behavior

By default, `run_benchmark` writes one CSV file per benchmark case under
`benchmark/ptdf/res/`. Each file is named after the PGLib case identifier, for
example: `benchmark/ptdf/res/pglib_opf_case9241_pegase.csv`

## Output File Structure

Each exported CSV contains the benchmark results for exactly one network case.
The file stores one row per benchmarked operation and configuration.

| Column Name      | Type      | Unit | Description                                                                       |
| ---------------- | --------- | ---- | --------------------------------------------------------------------------------- |
| `timestamp`      | `String`  | --   | Wall-clock time when the row was recorded.                                        |
| `case_name`      | `String`  | --   | Network case name.                                                                |
| `num_bus`        | `Int`     | --   | Number of buses.                                                                  |
| `num_branch`     | `Int`     | --   | Number of branches.                                                               |
| `backend`        | `String`  | --   | Backend, currently `cpu` or `cuda`.                                               |
| `device`         | `String`  | --   | CPU model name or CUDA device name used for the run.                              |
| `blas_threads`   | `Int`     | --   | BLAS thread count active during the benchmark run; mainly relevant on CPU.        |
| `ptdf_type`      | `String`  | --   | PTDF representation being benchmarked, either `full` or `lazy`.                   |
| `linear_solver`  | `String`  | --   | Linear solver used during PTDF construction.                                      |
| `operation`      | `String`  | --   | Benchmarked operation, either `construct` or `compute_flow`.                      |
| `rhs_width`      | `Int`     | --   | Batch width used for `compute_flow!`; this is `0` for construction rows.          |
| `min_time_ms`    | `Float64` | ms   | Minimum observed runtime across benchmark samples.                                |
| `median_time_ms` | `Float64` | ms   | Median observed runtime across benchmark samples.                                 |
| `mean_time_ms`   | `Float64` | ms   | Mean observed runtime across benchmark samples.                                   |
| `std_time_ms`    | `Float64` | ms   | Standard deviation of observed runtime across benchmark samples.                  |
| `memory_kb`      | `Float64` | KB   | Allocated memory reported by BenchmarkTools.                                      |