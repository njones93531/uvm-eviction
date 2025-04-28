#!/usr/bin/python3
import os
import re
import signal
import sh
import subprocess
import sys
import time

import config

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Experiment:

    def __init__(self, benchmark, benchmark_exe, benchmark_dir, psizes, policies,\
                 benchmark_arg_desc):
        self.benchmark = benchmark
        self.benchmark_exe = benchmark_exe
        self.benchmark_dir = benchmark_dir
        self.psizes = psizes
        self.policies = policies
          
    def get_benchmark_arg_desc(self, policy, psize):
        return f"{policy}_{psize}_{self.benchmark}"

    def __str__(self):
        return (f"Benchmark: {self.benchmark}\n"
                f"Benchmark Executable: {self.benchmark_exe}\n"
                f"Benchmark Directory: {self.benchmark_dir}\n"
                f"Psizes: {self.psizes}\n"
                f"Policies: {self.policies}\n")

    def _run(self, clear, psize, policy, timeout=float(os.environ['TIMEOUT'])): 
            print(f"Running experiment: {self.benchmark_dir}{self.get_benchmark_arg_desc(psize, policy)}")
           
            oldpwd = os.getcwd()
            os.chdir(self.benchmark_dir)

            #TODO fix this for benchmarks that don't use make explicitly
            print("Build benchmark")
            sh.make("-j")
            print("Starting execution")
            print(f"taskset 0xFFFFFFFF ./{self.benchmark_exe} {psize} -p {policy}")

            print(f"Setting environment variable TIMEOUT={timeout}")
            env = os.environ.copy()
            env['TIMEOUT'] = repr(timeout)
            start_time = time.monotonic()
            p = subprocess.Popen(["taskset", "0xFFFFFFFF", f'./{self.benchmark_exe}', str(psize), '-p', policy], stdout=subprocess.PIPE, preexec_fn=os.setsid, env=env)
            output, err = p.communicate()
            print("Waiting on process to die")
            p.wait()
            execution_time = time.monotonic() - start_time
            print(f"Execution time: {execution_time} seconds")

            print("out:", output)
            eprint("err:", err)

            os.chdir(oldpwd)
            print("Finished")
            return execution_time

    def run(self, clear=False):
        for psize in self.psizes:
            print(f"Starting Experiment Set {self.benchmark} {psize}")
            local_policies = self.policies.copy()
            # find baseline policy
            target = 'm' * len(local_policies[0])
            for policy in local_policies:
                if policy == target:
                    mm_policy = policy
                    local_policies.remove(policy)
                    break
            # run and time baseline first; set timeout for following policies based on this.
            baseline = self._run(clear, psize, policy)
            print(f"Experiment set {self.benchmark} {psize}: Measured baseline execution {target} at {baseline} seconds. Using {2 * baseline} as timeout for remaining experiments.")
            for policy in local_policies:
                self._run(clear, psize, policy, 2 * baseline)


#print("building warmup app")
#os.chdir(config.WARMUP_DIR)
#sh.make("-j")
#os.chdir(oldpwd)

if __name__ == "__main__":
    main()
