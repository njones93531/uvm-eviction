from experiments import Experiment

def get_cublas_experiments():
    benchmark = "stream"
    benchmark_exe = "cuda-stream"
    dir_prefix = "/home/najones/uvm-eviction/benchmarks/stream"
    psizes = ["125000704", "250001408", "375002112", "500002816", "625003520", "750004224"]
    mem_schemes = ["BCPinned", "APinned", "BCpAzc", "ApBCzc"]
    benchmark_dir = [dir_prefix]
    for s in mem_schemes:
        benchmark_dir.append(dir_prefix + "/" + s)
    benchmark_args = []
    for s in psizes:
        benchmark_args.append(["-s", s, "--float", "--triad-only"])
    kernel_args = {}
    benchmark_arg_desc = []
    for p in psizes: 
        benchmark_arg_desc.append("default_" + p)
    for s in mem_schemes: 
        for p in psizes:
            benchmark_arg_desc.append(s + "_" + p)
    kernel_version = "x86_64-460.27.04"
    kernel_variant = "vanilla"

    experiments = []

    for i in range(0, len(benchmark_dir)):
        for j in range(0, len(psizes)):
            experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir[i], benchmark_args[j], benchmark_arg_desc[len(psizes) * i + j],\
                    kernel_version, kernel_variant, kernel_args))
    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
