from experiments import Experiment
import sys

def get_cublas_experiments():
   
    #For strategies w/ size restrictions (i.e. placement) 
    bdir_restricted = ["AmovBmigCpin", "AmovBCmig", "AmovCmigBpin", "AmovBCpin"]
    psizes_restricted = [25000, 27500, 30000, 32500, 35000, 37500, 40000, 45000, 50000]
    bmark_args_restricted = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes_restricted]
    bmark_args_desc_restr=psizes_restricted

    #For strategies w/o size restrictions
    benchmark_dir = ["matmulABmigCpin", "matmulACmigBpin",  "matmulDef", "matmulApinBCmig", "matmulAmigBCpin", "matmulABCpin"]#"AmovBmigCpin", "AmovBCmig", "AmovCmigBpin", "AmovBCpin", "matmulDef", "matmulApinBCmig", "matmulAmigBCpin", "matmulABCpin"] 
    psizes = [25000, 27500, 30000, 32500, 35000, 37500, 40000, 45000, 50000, 55000, 60000]
    benchmark_args = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes]
    benchmark_arg_desc=psizes

    #For all 
    benchmark = "cublas"
    benchmark_exe = "matrixMul2"
    bdir_base = "/home/najones/uvm-eviction/benchmarks/matmul/"
    kernel_version = "x86_64-525.60.13"
    kernel_variant = ["vanilla"]
    kernel_args = [{}]
    experiments = []

    for i, ba in enumerate(bmark_args_restricted):
        for kv in kernel_variant:
            for kargs in kernel_args:
                for bdir in bdir_restricted:
                    experiments.append(Experiment(benchmark, benchmark_exe, bdir_base + bdir, ba, bdir + str(bmark_args_desc_restr[i]),\
                                    kernel_version, kv, kargs))

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
