from experiments import Experiment

def get_cublas_experiments():
    benchmark = "cublas"
    benchmark_exe = "matrixMul2"
    benchmark_dir = "/home/tnallen/dev/uvm-async-unmap/benchmarks/cublas"
    psizes = [4096*2, 4096*3, 4096 * 4, 4096*5, 4096 * 6, 4096*7, 4096 * 8]
    benchmark_args = [[f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"] for psize in psizes]
    benchmark_arg_desc=psizes
    kernel_version = "x86_64-460.27.04"
    #kernel_variant = ["batchd-asyncunmap"]
    kernel_variant = ["batchd", "batchd-asyncunmap"]
    kernel_args = [{}]#, {"uvm_perf_prefetch_enable": 0}]
    #kernel_args = [{}, {"uvm_perf_prefetch_enable": 0}]
    experiments = []
    for i, ba in enumerate(benchmark_args):
        for kv in kernel_variant:
            for kargs in kernel_args:
                experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, ba, benchmark_arg_desc[i],\
                                    kernel_version, kv, kargs))
    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
