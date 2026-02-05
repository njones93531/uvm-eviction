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
