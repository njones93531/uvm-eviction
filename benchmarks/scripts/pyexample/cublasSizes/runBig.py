from experiments import Experiment
import sys

def get_cublas_experiments():
    benchmark = "cublas"
    benchmark_exe = "matrixMul2"
    bdir_base = "/home/najones/uvm-eviction/benchmarks/matmul/"
    benchmark_dir = ["matmulABmigCpin", "matmulDef"]#, "matmulApinBCmig", "matmulAmigBCpin", "matmulABCpin"]#"AmovBmigCpin", "AmovBCmig", "AmovCmigBpin", "AmovBCpin", "matmulDef", "matmulApinBCmig", "matmulAmigBCpin", "matmulABCpin"] 
    psizes = [int(sys.argv[1])] #[45000,50000,55000,60000]#[25000, 27500, 30000, 32500, 35000, 37500, 40000]
    benchmark_args = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes]
    benchmark_arg_desc=psizes
    kernel_version = "x86_64-525.60.13"
    #kernel_variant = ["batchd-asyncunmap"]
    kernel_variant = ["vanilla"]
    kernel_args = [{}]#, {"uvm_perf_prefetch_enable": 0}]
    #kernel_args = [{}, {"uvm_perf_prefetch_enable": 0}]
    experiments = []
    for i, ba in enumerate(benchmark_args):
        for kv in kernel_variant:
            for kargs in kernel_args:
                for bdir in benchmark_dir:
                    experiments.append(Experiment(benchmark, benchmark_exe, bdir_base + bdir, ba, bdir + str(benchmark_arg_desc[i]),\
                                    kernel_version, kv, kargs))
    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
