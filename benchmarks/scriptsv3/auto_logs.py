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

LOGDIR = f"{BENCHMARK_DIR}/log_single"
KLOG = f"{LOGDIR}/spmm_klog.txt"

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
    os.chdir(config.SYSLOG_PATH)
    sh.make()
    os.chdir(oldpwd)
    process = subprocess.Popen([f"{config.SYSLOG_PATH}/{config.SYSLOG_EXE}", logfile])#, creationflags=subprocess.DETACHED_PROCESS)
    return process

def main():
    os.makedirs(LOGDIR, exist_ok=True)

    module_dir = os.path.expanduser(
        f"~{config.DRIVER_DIR}/{config.KERNEL_VERSION}/{config.KERNEL_VARIANT}/{config.KERNEL_LICENSE}"
    )
    print(module_dir)
    print("Building module")
    run(["make", "modules", "-j"], cwd=module_dir)

    load_uvm(f"{module_dir}/nvidia-uvm.ko", KERNEL_ARGS)

    #TODO add spmm-csr run
    #    collect metrics

    #at the end
    reset_uvm_module()

if __name__ == "__main__":
    main()
