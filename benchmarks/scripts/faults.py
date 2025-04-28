from experiments import Experiment
import math
import config

def get_experiment():
    psizes = ["9", "15"] #in GB
    benchmarks =['bfs-worst', 'cublas', 'GEMM', 'spmv-coo-twitter7', 'MVT', 'FDTD-2D', 'GRAMSCHM', 'stream']

                # ['2DCONV', 
                #  '2MM', 
                #  '3DCONV', 
                #  '3MM', 
                #  'ATAX', 
                #  'bfs-worst', 
                #  'BICG', 
                #  'cublas', 
                #  'FDTD-2D', 
                #  'GEMM', 
                #  'GESUMMV', 
                #  'GRAMSCHM', 
                #  'MVT', 
                #  'nw', 
                #  'spmv-coo', 
                #  'spmv-coo-twitter7', 
                #  'spmv-csr', 
                #  'stream']

    benchmark_exe = "run"
    benchmark_args = [[psize] for psize in psizes] 
    benchmark_dirs = [f"{config.BENCHDIR}/default/{bench}" for bench in benchmarks]
    kernel_version = "x86_64-535.104.05"
    kernel_variant = "faults"
    kernel_args = {}#,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
    experiments = []
    for j, psize in enumerate(psizes):
        for i, benchmark in enumerate(benchmarks):
            experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dirs[i], benchmark_args[j], f"{psize}_{benchmark}",\
                                        kernel_version, kernel_variant, kernel_args))
    return experiments

def main():
    experiments = get_experiment()
    for experiment in experiments:
        #experiment.print_exp()
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
