from experiments import Experiment

def get_cublas_experiments():
    benchmark = "stream"
    benchmark_exe = "cuda-stream"
    benchmark_dir = "/home/najones/uvm-eviction/benchmarks/stream"
    batch_sizes = [256, 512, 1024, 4096]
    benchmark_args = ["-s", "375002112", "--float", "--triad-only"]
    benchmark_arg_desc = []
    kernel_args = []
    trial_repeats = 1
    for gran in {"2m", "64k"}:
        for b in batch_sizes:
            kernel_args.append({"uvm_perf_access_counter_mimc_migration_enable": 1,"uvm_perf_access_counter_momc_migration_enable": 1,"uvm_perf_access_counter_granularity": gran, "uvm_perf_access_counter_threshold":1, "uvm_perf_access_counter_batch_count": b})
            for i in range(0, trial_repeats):
                benchmark_arg_desc.append("stream" + gran + "_batch-" + str(b))
    kernel_version = "x86_64-460.27.04"
    kernel_variant = "ac-tracking-full"

    experiments = []

    for i in range(0, len(benchmark_arg_desc)):
        experiments.append(Experiment(benchmark, benchmark_exe, benchmark_dir, benchmark_args, benchmark_arg_desc[i],\
                    kernel_version, kernel_variant, kernel_args[int(i/trial_repeats)]))
    return experiments

def main():
    experiments = []
    experiments += get_cublas_experiments()
    for experiment in experiments:
        experiment.run(clear=True)

if __name__ == "__main__":
    main()
