from experiments import Experiment

import config
import math
import os
import sys
import time
import argparse

def get_experiment(use_subset):
    psizes = config.PSIZES
    benchmarks = config.BENCHMARKS

    if use_subset: 
        psizes = config.PSIZES_SUBSET
        benchmarks = config.BENCHMARKS_SUBSET

    print("Beginning experiments")
    print(f"Experimental Benchmarks: {benchmarks}")
    print(f"Experimental Problem Sizes (GB): {psizes}")
    benchmark_exe = "run"
    benchmark_args = [psize for psize in psizes] 
    benchmark_dirs = [f"{config.BENCHDIR}/default/{bench}" for bench in benchmarks]
    experiments = []
    for i, benchmark in enumerate(benchmarks):
        experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dirs[i], psizes,\
                                    config.KERNEL_VERSION, config.KERNEL_VARIANT, config.KERNEL_ARGS))
    # benchmarks are making deep copies of this struct so this should be okay
    config.KERNEL_ARGS["uvm_perf_prefetch_enable"] = 0
    for i, benchmark in enumerate(benchmarks):
        experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dirs[i], psizes,\
                                      config.KERNEL_VERSION, config.KERNEL_VARIANT, config.KERNEL_ARGS))
    return experiments

def check_timeout():
    try:
        timeout = os.environ['TIMEOUT']
        timeout_value = float(timeout)  # Attempt to convert TIMEOUT to an integer
        print(f"TIMEOUT is set to a floating-point value: {timeout_value}")
    except KeyError:
        print("Error: The environment variable TIMEOUT is not set.")
        sys.exit(1)
    except ValueError:
        print(f"Error: TIMEOUT is set but is not an integer. Current value: {timeout}")
        sys.exit(1)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subset", action="store_true", help="Only compute a representative subset of data")

    args = parser.parse_args()

    check_timeout()
    experiments = get_experiment(args.subset)

    os.makedirs("slurm_out", exist_ok=True)
    with open("slurm_out/faults_benchmark_times.txt", "w") as f:
        for experiment in experiments:
            start_time = time.perf_counter_ns()
            experiment.run(clear=True)
            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time
            duration_sec = duration_ns / 1e9  # Convert nanoseconds to seconds
            f.write(f"Experiment {experiment}: {duration_sec:.9f} seconds\n")
            f.flush()


if __name__ == "__main__":
    main()

