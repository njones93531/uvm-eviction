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

def main():
    os.makedirs(LOGDIR, exist_ok=True)

    module_dir = os.path.expanduser(
        f"~{config.DRIVER_DIR}/{config.KERNEL_VERSION}/{config.KERNEL_VARIANT}/{config.KERNEL_LICENSE}"
    )
    print(module_dir)
    print("Building module")
    run(["make", "modules", "-j"], cwd=module_dir)

if __name__ == "__main__":
    main()
