#!/bin/bash -xe

#log="../../benchmarks/tyler-default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_21_GRAMSCHM/GRAMSCHM_klog"
#log="../../benchmarks/tyler-default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_18_GRAMSCHM/GRAMSCHM_klog"
log="../../benchmarks/tyler-default-complex/FDTD-2D/log_x86_64-555.42.02_faults-new_15_FDTD-2D/FDTD-2D_klog"

ulimit -v 230000000
output_file=`python3 fault_parsing.py $log metrics`
#if [ ! -e $output_file ]; then
    python3 metric_plot.py $log
#else
#    echo "skipping $output_file because it already exists."
#fi


