from experiments import Experiment

def get_experiment():
    benchmark = "cublas"
    benchmark_exe = "matrixMul2"
    benchmark_dir = "/home/najones/uvm-eviction/benchmarks/matmul"
    psize = 4096*2
    benchmark_args = [f"-hA={psize}", f"-hB={psize}", f"-wA={psize}", f"-wB={psize}"]
    benchmark_arg_desc="2m_cublas2"
    kernel_version = "x86_64-460.27.04"
    kernel_variant = "ac-tracking-full"
    #kernel_variant = ["batchd", "batchd-asyncunmap"]
    kernel_args = {"uvm_perf_access_counter_granularity": "2m", "uvm_perf_access_counter_threshold": 1}
    #kernel_args = [{}, {"uvm_perf_prefetch_enable": 0}]
    experiments = Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args, benchmark_arg_desc,\
                                    kernel_version, kernel_variant, kernel_args)
    return experiments

def main():
    experiment = get_experiment()
    experiment.run(clear=True)

if __name__ == "__main__":
    main()
