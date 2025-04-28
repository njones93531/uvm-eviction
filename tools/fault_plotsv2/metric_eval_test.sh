#!/bin/bash -xe

#log="../../benchmarks/eval-apps/conjugateGradientUM/log_x86_64-555.42.02_faults-new_15_conjugateGradientUM/conjugateGradientUM_klog"
#log="../../benchmarks/eval-apps/hpgmg/log_x86_64-555.42.02_faults-new_15_hpgmg/hpgmg_klog"
#log="../../benchmarks/eval-apps/tealeaf/log_x86_64-555.42.02_faults-new_15_tealeaf/tealeaf_klog"
log="../../benchmarks/eval-apps/spmv-coo-twitter7/log_x86_64-555.42.02_faults-new_1672_spmv-coo-twitter7/spmv-coo-twitter7_klog"

ulimit -v 230000000
output_file=`python3 fault_parsing.py $log metrics_stats_relative_eval`
#if [ ! -e $output_file ]; then
    python3 metric_plot.py $log -o metrics_stats_relative_eval
#else
#    echo "skipping $output_file because it already exists."
#fi


