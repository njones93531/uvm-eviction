#!/usr/bin/python3
import os
import re
import sh
import subprocess
import sys
import time

import config

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_module(module, args={}):
    # load and unload module first to make sure its unloaded w/o error
    sh.sudo("modprobe", module)
    sh.sudo("rmmod", "-f", module)

    print(f"modprobe {module}", args)
    arg_strs = [f"{key}={value}" for key, value in args.items()] 
    sh.sudo("modprobe", module, *arg_strs)
    if config.WARMUP_ENABLED:
        p = subprocess.Popen([f'{config.WARMUP_DIR}/{config.WARMUP_EXE}'], stdout=subprocess.PIPE)
        # insert warmup error logging?
        p.wait()

def install_module(path):
    print(f"Installing kernel module at {path}")
    oldpwd = os.getcwd()
    os.chdir(path)
    sh.make("modules", "-j")
    sh.sudo("make", "modules_install", "-j")
    os.chdir(oldpwd)

def sys2csv(logfile):
    subprocess.Popen([f"{config.SYS2CSV_PATH}/{config.SYS2CSV_SCRIPT}", logfile]).wait()

def init_syslogger(logfile):
    sh.sudo("dmesg", "-c")
    process = subprocess.Popen([f"{config.SYSLOG_PATH}/{config.SYSLOG_EXE}", logfile])#, creationflags=subprocess.DETACHED_PROCESS)
    time.sleep(8)
    return process

def close_syslogger(process):
    process.kill()
    
kernel_arg_dict={"uvm_perf_access_counter_granularity": "gran", "uvm_perf_access_counter_threshold": "thold", "uvm_perf_access_counter_mimc_migration_enable": "mimc", "uvm_perf_access_counter_momc_migration_enable": "momc"}
def kernel_args_string(kernel_args):
    if len(kernel_args.keys()) == 0:
        return ""
    return "_".join([kernel_arg_dict[key] for key in kernel_args.keys()]) + "_"

class Experiment:

    def __init__(self, benchmark, benchmark_exe, benchmark_dir, benchmark_args,\
                 benchmark_arg_desc, kernel_version, kernel_variant, kernel_args):
        self.benchmark = benchmark
        self.benchmark_exe = benchmark_exe
        self.benchmark_dir = benchmark_dir
        self.benchmark_args = benchmark_args
        self.benchmark_arg_desc = benchmark_arg_desc
        self.kernel_version = kernel_version
        self.kernel_variant = kernel_variant
        self.kernel_args = kernel_args
            

    def run(self, clear=False):
        logdir_base = f"{self.kernel_version}_{self.kernel_variant}_{kernel_args_string(self.kernel_args)}{self.benchmark_arg_desc}"
        logdir = f"{self.benchmark_dir}/log_{logdir_base}"
        klog = f"{logdir}/{logdir_base}_klog"
        plog = f"{logdir}/{logdir_base}_perf.txt"
        print("Running experiment:", logdir_base)
        if not clear and os.path.exists(klog) and os.path.exists(plog):
            print("Existing experiment log found")
            return
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if os.path.exists(klog):
            os.remove(klog)
        if os.path.exists(plog):
            os.remove(plog)

        install_module(f"{config.DRIVER_DIR}/{self.kernel_version}/{self.kernel_variant}/kernel/")
        load_module("nvidia_uvm", self.kernel_args)

        slog_proc = init_syslogger(klog)
        oldpwd = os.getcwd()
        os.chdir(self.benchmark_dir)

        #TODO fix this for benchmarks that don't use make explicitly
        print("Build benchmark")
        sh.make("-j")
        print("Starting execution")
        p = subprocess.Popen(["taskset", "-c", "4,5", f'./{self.benchmark_exe}'] + self.benchmark_args, stdout=subprocess.PIPE)
        output, err = p.communicate()
        exit_code = p.wait()
        close_syslogger(slog_proc)
        sys2csv(klog)

        print("out:", output)
        eprint("err:", err)
        with open(plog, "w+") as of:
            for line in output.decode().split("\n"):
                if re.search("alloced,", line) or re.search("perf,", line):
                    of.write(line + "\n")

        os.chdir(oldpwd)
        print("Finished", logdir_base)

print("Building logger exe")
oldpwd = os.getcwd()
os.chdir("/home/najones/uvm-eviction/tools/syslogger/")
sh.make("-j")
os.chdir(oldpwd)
#print("building warmup app")
#os.chdir(config.WARMUP_DIR)
#sh.make("-j")
#os.chdir(oldpwd)

if __name__ == "__main__":
    main()
