import itertools
import math

import numpy as np

from experiments_nolog import Experiment

def get_combos(a):
    options = ['m', 'h']
    strats = []
    while len(strats) < len(options)**a:
        strat = [options[len(strats) // len(options)**i % len(options)] \
                for i in range(0, a)]
        strats.append(strat)
    return strats

def zip_strat(a, b, default):
    print(f"a, b: {a}, {b}")
    count = 0
    for i, c in enumerate(a):
        if c == '-':
            a[i] = default
        else:
            a[i] = b[count]
            count = count + 1
    return a

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
    
    aois =  {"bfs-worst":0,
            "spmv-coo-twitter7":0,#-1,
            "cublas":-1,
            "2DCONV":0,
            "2MM":-1,
            "3DCONV":0,
            "3MM":-1,
            "ATAX":-1,
            "BICG":0,
            "FDTD-2D":1,
            "GEMM":-1,
            "GESUMMV":-1,
            "GRAMSCHM":-1,
            "MVT":-1,
            "nw":0,
            "stream":-1,
    }

    fixed_strats = {"bfs-worst":[],
            "spmv-coo-twitter7":[],#-1,
            "cublas":[],
            "2DCONV":[],
            "2MM":[],
            "3DCONV":[],
            "3MM":[],
            "ATAX":[],
            "BICG":[],
            "FDTD-2D":[0],
            "GEMM":[],
            "GESUMMV":[2,3,4],
            "GRAMSCHM":[],
            "MVT":[1,2,3,4],
            "nw":[],
            "stream":[],
    }

    benchmark_exe = "run_numa.sh"
    kernel_version = "x86_64-555.42.02"
    #kernel_version = "x86_64-535.104.05"
    kernel_variant = "vanilla"
    experiments = []
    kernel_args = {}#,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
 
    for benchmark in benchmarks:#, "cublas", "bfs-worst", "GRAMSCHM", "GEMM"]:
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
