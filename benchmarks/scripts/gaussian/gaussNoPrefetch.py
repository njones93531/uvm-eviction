from experiments import Experiment
import sys

#Experiment to see if prefetching turned off makes Gaussian work better
#It performed terribly.

def get_cublas_experiments():
   
    #For strategies already done 
    r_policies  = ["mmm"]
    psizes_restricted = [40000, 50000, 10]
    benchmark_args_restricted = [[["-s", str(psize), "-q", "-p", policy] for policy in r_policies] for psize in psizes_restricted]
    bmark_args_desc_restr=[[f"{policy}_{psize}" for policy in r_policies] for psize in psizes_restricted]

    #For all 
    benchmark = "gaussian"
    benchmark_exe = "gaussian"
    bdir_base = "/home/najones/uvm-eviction/benchmarks/UVMBench/gaussian"
    kernel_version = "x86_64-525.60.13"
    kernel_variant = "vanilla"
    kernel_args = [{"uvm_perf_prefetch_enable": 0}]
    experiments = []

    for i in range(0, len(benchmark_args_restricted)):
        for j, ba in enumerate(benchmark_args_restricted[i]):
            experiments.append(Experiment(benchmark, benchmark_exe, bdir_base, ba, str(bmark_args_desc_restr[i][j]), kernel_version, kernel_variant, kernel_args[0]))

    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
