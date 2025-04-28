from experiments import Experiment
import argparse
import os
import time
import config 

kernel_version = "x86_64-535.104.05"
kernel_variant = ["eviction"]

granularities = {"2m", "64k"}
tholds = [1, 4, 8, 16, 32]
kernel_args = [{"uvm_perf_access_counter_mimc_migration_enable": 1,"uvm_perf_access_counter_momc_migration_enable": 1,\
                "uvm_perf_access_counter_granularity": gran, "uvm_perf_access_counter_threshold": t}\
                for gran in granularities for t in tholds]

def get_experiments(benchmark):
    benchmark_dir = f"{config.BENCHDIR}/{benchmark}"
    if benchmark == "matmul":
        benchmark_exe = "matmul"
        psizes = [i*4096 for i in range(1, 4)]
        benchmark_args = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes]
        benchmark_arg_desc=psizes
    elif benchmark == "stream":
        benchmark_exe = "cuda-stream"
        psizes = [125000704*i for i in range(1, 4)]

        benchmark_args = [["-s", f"{psize}", "--float", "--triad-only"] for psize in psizes]
        benchmark_arg_desc=psizes


    experiments = []
    for kv in kernel_variant:
        for kargs in kernel_args:
            for i, ba in enumerate(benchmark_args):
                experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, ba, benchmark_arg_desc[i],\
                                    kernel_version, kv, kargs))
    return experiments

#TODO add arg for clear, iters
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', '-c', action='store_true', help='clear files')
    args = parser.parse_args()

    experiments = []
    for bench in ["matmul", "stream"]:
        experiments += get_experiments(bench)

    if (args.clear):
        print("Clearing past experiment files because of CLI flag; pausing 30 seconds for cancel")
        time.sleep(30)
    for experiment in experiments:
        print(experiment)
        experiment.run_exp(clear=args.clear, iters=5)

if __name__ == "__main__":
    main()
