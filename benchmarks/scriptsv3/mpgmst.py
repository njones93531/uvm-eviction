import itertools
import math

import numpy as np

from experiments_nolog import Experiment

def get_experiments():
    psizes = [9, 12, 15, 18, 21] #For every experiment, fill the device memory entirely
    benchmarks = [\
                    #"GEMM",                 #3
                    "cublas",                #3
                    "stream",                   #2
                    #"spmv-coo-twitter7",       #5
                    "bfs-worst",            #4
#                    "2DCONV",               #2
#                    "2MM",                  #5
#                    "3DCONV",               #2
#                    "3MM",                  #7
#                    "ATAX",                 #4
#                    "BICG",                 #5
#                    "GESUMMV",              #5
                    "GRAMSCHM",             #3
                    "FDTD-2D",              #4
#                    "MVT",                   #5
#                    "nw",
                     ]

    allocs = {"bfs-worst":3,
            "spmv-coo-twitter7":5,
            "cublas":3 ,
            "2DCONV":2,
            "2MM":5,
            "3DCONV":2,
            "3MM":7,
            "ATAX":4,
            "BICG":5,
            "FDTD-2D":3,
            "GEMM":3,
            "GESUMMV":5,
            "GRAMSCHM":3,
            "MVT":5,
            "nw":2,
            "stream":3,
    }
    
    bdir_base = "../strategied/"
    bdir_dict = {}
    for bmark in benchmarks: 
        bdir_dict[bmark] = f"{bdir_base}{bmark}/"

    benchmark_exe = "run_numa.sh"
    kernel_version = "x86_64-555.42.02"
    kernel_variant = "vanilla"
    experiments = []
    kernel_args = {}#,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
 
    for benchmark in benchmarks:
        benchmark_dir = bdir_dict[benchmark]
        kernel_args = {}
        nallocs = allocs[benchmark]

        #assemble policy list
        policy_list = [''.join(combination) for combination in itertools.product(['m', 'h', 'd'], repeat=nallocs)]
        experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, psizes, policy_list, benchmark_dir,\
                                      kernel_version, kernel_variant, kernel_args))

    return experiments

def main():
    experiments = get_experiments()
    for experiment in experiments:
        #experiment.print_exp()
        print(experiment)
        experiment.run()

if __name__ == "__main__":
    main()
