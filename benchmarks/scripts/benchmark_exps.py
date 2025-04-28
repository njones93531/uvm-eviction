from experiments import Experiment
import math

def get_experiment():
    benchmark = "cublas"
    benchmark_exe = "matrixMul2"
    benchmark_dir = "/home/jsasser4/CLionProjects/uvm-eviction/benchmarks/matmul/matmulABCpin"
    psizes = [4096] #IPDPS size and 4096 * 9 1.25x oversub
    benchmark_args = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes]
    benchmark_arg_desc=[f"cublas_{psize}" for psize in psizes]
    kernel_version = "x86_64-535.104.05"
    kernel_variant = "eviction"
    kernel_args = {"uvm_perf_access_counter_mimc_migration_enable":1,"uvm_perf_access_counter_momc_migration_enable":1,"uvm_perf_access_counter_granularity":"2m","uvm_perf_prefetch_enable":0, "uvm_perf_access_counter_threshold":"1"}
    experiments = []
    for i, psize in enumerate(psizes):
        experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args[i], benchmark_arg_desc[i],\
                                    kernel_version, kernel_variant, kernel_args))
    return experiments

def main():
    experiments = get_experiment()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
