from experiments import Experiment

def get_experiment():
    benchmark = "touch_pages"
    benchmark_exe = "touch_pages"
    benchmark_dir = "/home/najones/uvm-eviction/demo/access_counters"
    #psize = 4096*2
    benchmark_args = []
    benchmark_arg_desc="touch64k-8t"
    kernel_version = "x86_64-460.27.04"
    kernel_variant = "ac-tracking-full"
    #kernel_variant = ["batchd", "batchd-asyncunmap"]
    kernel_args = {"uvm_perf_access_counter_mimc_migration_enable": 1,"uvm_perf_access_counter_momc_migration_enable": 1,"uvm_perf_access_counter_granularity": "64k", "uvm_perf_access_counter_threshold": 8}
    #kernel_args = [{}, {"uvm_perf_prefetch_enable": 0}]
    experiments = Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args, benchmark_arg_desc,\
                                    kernel_version, kernel_variant, kernel_args)
    return experiments

def main():
    experiment = get_experiment()
    experiment.run(clear=True)

if __name__ == "__main__":
    main()
