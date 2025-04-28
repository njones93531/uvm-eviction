from experiments_nolog import Experiment
import math
import numpy as np

def get_combos(a):
    options = ['m', 'h']
    strats = []
    while len(strats) < len(options)**a:
        strat = [options[len(strats) // len(options)**i % len(options)] \
                for i in range(0, a)]
        strats.append(strat)
    return strats

def zip_strat(a, b, default):
    count = 0
    for i, c in enumerate(a):
        if c == '-':
            a[i] = default
        else:
            a[i] = b[count]
            count = count + 1
    return a

def get_experiment():
    psizes = [18, 21] #For every experiment, fill the device memory entirely
    benchmarks = ["spmv-coo-twitter7",       #5
                    "bfs-worst",            #4
                    "cublas",                #3
                    "2DCONV",               #2
                    "2MM",                  #5
                    "3DCONV",               #2
                    "3MM",                  #7
                    "ATAX",                 #4
                    "BICG",                 #5
                    "FDTD-2D",              #4
                    "GEMM",                 #3
                    "GESUMMV",              #5
                    "GRAMSCHM",             #3
                    "MVT",                   #5
                    "nw",
                    "stream"]                   #2

    allocs = {"bfs-worst":3,
            "spmv-coo-twitter7":5,
            "cublas":3 ,
            "2DCONV":2,
            "2MM":5,
            "3DCONV":2,
            "3MM":7,
            "ATAX":4,
            "BICG":5,
            "FDTD-2D":4,
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

    benchmark_exe = "run"
    kernel_version = "x86_64-535.104.05"
    kernel_variant = "vanilla"
    experiments = []
    kernel_args = {}#,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
    #for sgemm, we are interested in every allocation
   
#    #all the mmm because I messed it up 
#    for benchmark in ["cublas", "3MM"]:
#        benchmark_dir = bdir_dict[benchmark]
#        pressure = 0
#        thold = 50
#        
#        for pressure in [0, 0.5, 0.80, 0.90, 0.95, 0.99]:
#            thold = 50
#            #add the mmm policy 
#            policy = ['m' for i in range(0, allocs[benchmark])]
#            kernel_args = {}
#            aoi = 0 #shows a policy that applies to all aoi
#            benchmark_args = [f"{psize}", "-p", "".join(policy), "-aoi", f"{aoi}", "-r", f"{pressure}"]
#            experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args,\
#                    f"pressure_"+"".join(policy)+f"_pressure_{pressure}_aoi_{aoi}_thold_{thold}_{benchmark}",\
#                    kernel_version, kernel_variant, kernel_args))
# 
 
    for benchmark in benchmarks:
        benchmark_dir = bdir_dict[benchmark]
        kernel_args = {}

        #assemble policy list
        policy_list = []
        #done = {}
#        done = {"mmmmm", "dmmmm", "hmmmm", "mdmmm", "ddmmm", "hdmmm", "mhmmm", "dhmmm", "hhmmm", "mmdmm", "dmdmm", "hmdmm", "mddmm", "dddmm", "hddmm", "mhdmm", "dhdmm", "hhdmm", "mmhmm", "dmhmm", "hmhmm", "mdhmm", "ddhmm", "hdhmm", "mhhmm", "mdhdm", "mhhhm", "mdmhh", "hmdhh", "ddmmh", "hdmdm", "mmhhd", "dddhh", "mhhmh", "mdhhm", "dhddm", "hhmmh", "hdddh", "hddmd", "hhhhd", "hmdhm", "mmddm", "mmhhm", "dhdhd", "hhddd", "hmmmh", "hmmhd", "hhhdh", "dmmdm", "hdhhh", "hdmhd", "mmdhd", "dhhhh", "mdmmh", "hmdmd", "hhdhm", "mmhmh", "dmdhd", "hdhhd"}
        nallocs = allocs[benchmark]
        policy_list_3_alloc = []
        mvt_policy_list = ["hmmmm", "mmmmm", "dmmmm"]
        base_strat = ['d' for i in range(0, nallocs)]
        for i in fixed_strats[benchmark]:
            base_strat[i] = '-'
        #filler = get_combos(nallocs - len(fixed_strats[benchmark]))
        filler = ["hmm", "mhm", "mmh", "mmm"]
        for combo in filler:
            combo = list(combo) #unnecessary when using get_combos()
            strat = zip_strat(base_strat.copy(), combo, 'd')
            #policy_list.append("".join(strat))
            policy_list_3_alloc.append("".join(strat))
        #policy_list = set(mvt_policy_list)
        if benchmark != "MVT":
            policy_list = set(policy_list_3_alloc)
        #policy_list = policy_list - done
        for psize in psizes:
            for policy in policy_list:
                benchmark_args = [f"{psize}", "-p", policy]
                experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args,\
                f"{policy}_{psize}_{benchmark}",\
                kernel_version, kernel_variant, kernel_args))

    return experiments

def main():
    experiments = get_experiment()
    for experiment in experiments:
        #experiment.print_exp()
        experiment.run()

if __name__ == "__main__":
    main()
