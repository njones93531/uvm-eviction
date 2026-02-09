#!/usr/bin/env python3

import os
import subprocess
import time
import sys
import sh

import config

BENCHMARK_DIR = "../default/spmm-csr"
BENCHMARK_EXE = "./spmm_csr"
KERNEL_ARGS = {}

def run(cmd, cwd=None):
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)

def load_uvm(module_path, args):
    print("Unloading nvidia-uvm (if loaded)")
    try:
        sh.sudo("rmmod", "-f", "nvidia-uvm")
    except Exception:
        pass

    arg_list = [f"{k}={v}" for k, v in args.items()]
    print("Loading nvidia-uvm with args:", arg_list)
    sh.sudo("insmod", module_path, *arg_list)

def reset_uvm_module():
    try:
        subprocess.run(
            ["sudo", "rmmod", "-f", "nvidia-uvm"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        print("Warning: rmmod failed:", e)

    time.sleep(1)

    result = subprocess.run(
        ["sudo", "modprobe", "nvidia-uvm"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("ERROR: modprobe nvidia-uvm failed")
        print(result.stderr)
        raise RuntimeError("Failed to reset nvidia-uvm module")

    print("nvidia-uvm reset")

def get_device_number(device_name):
    with open("/proc/devices") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == device_name:
                return int(parts[0])
    return None

def init_syslogger(logfile):
    # this is required for voltron because it doesn't support kernel-open and udev has to create this file, 
    # which has a slight asynchronous delay
    dev_num = get_device_number("hpcs_logger")
    counter = 0
    while not os.path.exists("/dev/hpcs_logger"):
        counter = counter + 1
        sh.sudo("mknod", "-m", "0666", "/dev/hpcs_logger", "c", str(dev_num), "0")
        if counter > 10:
            print("/dev/hpcs_logger still does not exist after 10 seconds; check dmesg for errors")
            sys.exit(1)
        time.sleep(1)
    oldpwd = os.getcwd()
    os.chdir(os.path.expanduser(f"~{config.SYSLOG_PATH}"))
    sh.make()
    os.chdir(oldpwd)
    exe = os.path.expanduser(f"~{config.SYSLOG_PATH}/{config.SYSLOG_EXE}")
    process = subprocess.Popen([exe, logfile])#, creationflags=subprocess.DETACHED_PROCESS)
    return process

def run_spmm(arg_set):
    print("Build benchmark")
    os.chdir(f"{BENCHMARK_DIR}")
    print(os.getcwd())
    sh.make("-j")
    print("Starting execution") 
    
    cmd = ["taskset", "0xFFFFFFFF", BENCHMARK_EXE] + [str(a) for a in arg_set]
    print("Running benchmark:", cmd)
    subprocess.run(cmd, check=True)
   
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print(f"Running command {cmd}")
    output, err = p.communicate()
    exit_code = p.wait()
    
    print("out:", output)
    print("err:", err)
    print("exit code:", exit_code)

def main():

    module_dir = os.path.expanduser(
        f"~{config.DRIVER_DIR}/{config.KERNEL_VERSION}/{config.KERNEL_VARIANT}/{config.KERNEL_LICENSE}"
    )
    print(module_dir)
    print("Building module")
    run(["make", "modules", "-j"], cwd=module_dir)

    load_uvm(f"{module_dir}/nvidia-uvm.ko", KERNEL_ARGS)

    arg_sets = [
        [0.1,   0.1,   1],
    ]

    benchmark="spmm-csr"
    logdir = f"{BENCHMARK_DIR}/{config.KERNEL_VERSION}_{config.KERNEL_VARIANT}_{arg_sets[0][0]}_{benchmark}"
    klog = f"{logdir}/klog_{benchmark}.txt"
    os.makedirs(logdir, exist_ok=True)

    #TODO add spmm-csr run
    #    collect metrics
    slog_proc = init_syslogger(klog)

    size = run_spmm(arg_sets[0])

    #at the end
    slog_proc.kill()
    reset_uvm_module()

if __name__ == "__main__":
    main()
