# Data Collection
## Definitions
- `benchmarks` includes the following:
  - bfs-worst
  - cublas
  - FDTD-2D
  - GEMM
  - GRAMSCHM
  - MVT
  - spmv-coo-twitter7
  - stream

## Data Requirements 
Metric generation will require the following data: 
1. Memory access log data
  - For each benchmark in `benchmarks`
  - For each oversubscription ratio in:
    - 0.75x
    - 1.25x 
  - For one or more prefetching modes in: 
    - prefetching
    - no prefetching
2. Memory pressure performance data
  - For each benchmark in `benchmarks` (except `spmv-coo-twitter`)
  - For each oversubscription ratio in: 
    - 0.75x
    - 1.00x
    - 1.25x
    - 1.50x
    - 1.75x
  - For each allocation placement in:
    - pinned to host
    - not pinned
  - For each allocation of interest (dependent on application)
  - For one or more prefetching modes in: 
    - prefetching
    - no prefetching

## Collecting Data
*Prerequisite:* Open `uvm-eviction/benchmarks/scripts/config.py` and ensure
   that `ROOT` is set properly to the absolute path of `uvm-eviction`.

**Data 1** can be collected using scripts in `uvm-eviction/benchmarks/scripts/`.
  Steps: 
  1. Open `faults.py`. Ensure that: 
    - `psizes` is set to `[<x1>, <x2>...]` where `<x1>` etc are the desired 
      problem sizes in GB, as strings. These should match the oversubscription 
      ratios desired.
    - `benchmarks` is set to `[<bmark1>, <bmark2> ...]`.
    - `kernel_variant` is set to `faults`
    - `kernel_args` is set properly:
      - `{}` for prefetching enabled
      - `{"uvm_perf_prefetch_enable":0}` for prefetching disabled
  2. Open `genfaults.sh`. Ensure that: 
    - (If using slurm) The slurm arguments match your preference.
    - The script properly loads `cuda` and `gcc`. 
    - The `TIMEOUT` is set to your preference. Each experiment will be stopped
      after `TIMEOUT` seconds if it does not end before. `TIMEOUT` must be set.
  3. Run `genfaults.sh` using `sbatch genfaults.sh` or `./genfaults.sh`.
 
**Data 2** can be collected using scripts in `uvm-eviction/benchmarks/scripts/`.
  Steps:
  1. Open `mpgmst.py`. Ensure that:
    - `psizes` is set to `[<x1>, <x2>...]` where `<x1>` etc are the desired 
      problem sizes in GB, as strings. These should match the oversubscription 
      ratios desired.
    - `benchmarks` is set to `[<bmark1>, <bmark2> ...]`.
    - `kernel_variant` is set to `vanilla`
    - `kernel_args` is set properly:
      - `{}` for prefetching enabled
      - `{"uvm_perf_prefetch_enable":0}` for prefetching disabled
  2. Open `sbatchmempressall.sh`. Ensure that: 
    - (If using slurm) The slurm arguments match your preference.
    - The script properly loads `cuda` and `gcc`. 
    - The `TIMEOUT` is set to your preference. Each experiment will be stopped
      after `TIMEOUT` seconds if it does not end before. `TIMEOUT` must be set.
  3. Run `sbatchmempressall.sh` using `sbatch sbatchmempressall.sh` or `./sbatchmempressall.sh`. 

## Processing Data 

The following outputs can be generated using the data described above: 
 |Output         | Data Required|
 |---------------|------------- |
 | fault plots   |  1           |
 | violin plots  |  1           |
 | scatter plots |  1           |
 | vs plots      |  1           |
 | mempress      |  2           |

Unless stated otherwise, all of these can be generated from scripts in 
`/uvm-eviction/tools/fault_plots/`.

**Fault plots:**
1. Open `mass_fault_plots.sh`. Ensure that:
  - (If using slurm) slurm arguments match your preference
  - script properly loads `julia`
  - `threshold` is set to preference. `threshold` marks the maximum number of
    lines that can be processed without `OOM` errors. Only the first
    `threshold` lines will be processed, the rest are ignored.
  - all preferred benchmarks are listed
  - all preferred psizes are listed
  - ONLY preferred methods are listed
    - "" for prefetching
    - "\_nopf" for no prefetching
2. Run `mass_fault_plots.sh` using `sbatch mass_fault_plots.sh` or `./mass_fault_plots.sh`. 

**Violin plots:**

There are two types of violin plots. Violin plots for each benchmark are generated using
`density.py`; see the instructions for scatter plots. Violin comparison plots
are generated using `conclusion_five.sh`. See instructions below. 
1. Open `conclusion_five.sh`. Ensure that:
  - ONLY preferred methods are listed
    - "" for prefetching
    - "\_nopf" for no prefetching
  - `input` is set to the desired input file
2. Open the desired input file, hereafter called `input`. Ensure that:
  - `input` contains the absolute paths of exactly 5 log files, one per line
    and no other content. 
3. Run `./conclusion_five.sh`.

**Scatter plots:**
1. Open `mass_density_plots.sh`. Ensure that:
  - all preferred benchmarks are listed
  - all preferred psizes are listed
  - ONLY preferred methods are listed
    - "" for prefetching
    - "\_nopf" for no prefetching
  - `bench` is set to a valid path for each benchmark
2. Open `density.py`. Ensure that:
  - `max_input_bytes` is set appropriately. This value is the maximum amount
    of data `density.py` will read from an input file to process, the rest
    of the file will be ignored. Prevents `OOM` errors. 
  - in `main()`, ensure that one or more of the following tasks are 
    uncommented, as desired:
    - scatter\_plot\_polyfit
    - violin\_plot\_all\_metrics
    - scatter\_plot\_all\_metrics
3. Run `./mass_density_plots.sh`.

**vs plots:** 
1. Open `mass_fault_csv.sh`. Ensure that:
  - `output` is set as desired
  - all preferred benchmarks are listed
  - all preferred psizes are listed
  - `bench` is set to a valid log data path for each benchmark
2. Run `./mass_fault_csv.sh`. This will create an `output` csv file.
3. Open `csv_vs_plots_all.sh`. Ensure that: 
  - `x` is the desired x column
  - `y` is the desired y column (or `all`)
  - all preferred benchmarks are listed
4. Run `./csv_vs_plots_all.sh <infile>`. This will genereate vs plots from the 
  provided file.

**mempressure plots**
1. Go to `/uvm-eviction/benchmarks/strategied/common/plot/`.
2. Open `mempress_plotall.sh`. Ensure that
  - all preferred benchmarks are listed
  - `bench` is set to a valid mempress data path for each benchmark
3. Run `./mempress_plotall.sh`. 

