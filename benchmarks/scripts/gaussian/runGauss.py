from experiments import Experiment
import sys

def get_cublas_experiments():
   
    #For strategies already done 
    #r_policies  = ["mmm","mpm","mpp","mmp","ppm","pmm","ppp","pmp"] #['ppp','mmm','pmm','mpm']
    #psizes_restricted = [10000,15000, 200 30000, 50000]
    #benchmark_args_restricted = [[["-s", str(psize), "-q", "-p", policy] for policy in r_policies] for psize in psizes_restricted]
    #bmark_args_desc_restr=[[f"{policy}_{psize}" for policy in r_policies] for psize in psizes_restricted]

    #For strategies w/o size restrictions
    policies = ["mmm","mmp","mpm","pmm","mpp","ppm","ppp","pmp","lll","llp","lpl","pll","ppl","plp","lpp","mpl","mlp","lpm","lmp","plm","pml"]
    psizes = [10000,12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000]
    benchmark_args = [[["-s", str(psize), "-q", "-p", policy] for policy in policies] for psize in psizes]
    benchmark_arg_desc=[[f"{policy}_{psize}" for policy in policies] for psize in psizes]


    #For all 
    benchmark = "gaussian"
    benchmark_exe = "gaussian"
    bdir_base = "/home/najones/uvm-eviction/benchmarks/UVMBench/gaussian"
    kernel_version = "x86_64-525.60.13"
    kernel_variant = "vanilla"
    kernel_args = [{}]
    experiments = []

    #for i in range(0, len(benchmark_args_restricted)):
     #   for j, ba in enumerate(benchmark_args_restricted[i]):
      #      experiments.append(Experiment(benchmark, benchmark_exe, bdir_base, ba, str(bmark_args_desc_restr[i][j]), kernel_version, kernel_variant, kernel_args[0]))

    for i in range(0, len(benchmark_args)):
        for j, ba in enumerate(benchmark_args[i]):
            experiments.append(Experiment(benchmark, benchmark_exe, bdir_base, ba, str(benchmark_arg_desc[i][j]), kernel_version, kernel_variant, kernel_args[0]))

    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
