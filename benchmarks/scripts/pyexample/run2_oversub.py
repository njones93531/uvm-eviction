from experiments import Experiment

def get_experiment():
    benchmark = "oversub"
    benchmark_exe = "uvm_oversubs"
    benchmark_dir = "/home/najones/uvm-eviction/demo/unified-memory-oversubscription"
    psize = 4096*2
    benchmark_args = ["-p", "2.0", "-a", "block_streaming", "-m", "zero_copy"]
    kernel_version = "x86_64-460.27.04"
    kernel_variant = "ac-tracking-full"
    #kernel_variant = ["batchd", "batchd-asyncunmap"]
    kernel_args = {"uvm_perf_access_counter_granularity": "2m", "uvm_perf_access_counter_threshold": 1}
    #kernel_args = [{}, {"uvm_perf_prefetch_enable": 0}]
    benchmark_arg_desc="2m_oversub2"
    experiments = Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args, benchmark_arg_desc,\
                                    kernel_version, kernel_variant, kernel_args)
    return experiments

def main():
    experiment = get_experiment()
    experiment.run(clear=True)

if __name__ == "__main__":
    main()
