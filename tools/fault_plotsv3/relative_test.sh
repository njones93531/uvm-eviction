#!/bin/bash -xe

#log="../../benchmarks/default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_21_GRAMSCHM/GRAMSCHM_klog"
#log="../../benchmarks/default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_18_GRAMSCHM/GRAMSCHM_klog"
#log="../../benchmarks/default/FDTD-2D/log_x86_64-555.42.02_faults-new_15_FDTD-2D/FDTD-2D_klog"
log="../../benchmarks/default/sgemm/log_x86_64-555.42.02_faults-new_15_sgemm/sgemm_klog"
log="../../benchmarks/default/sgemm/log_x86_64-555.42.02_faults-new_12_sgemm/sgemm_klog"
#log="../../benchmarks/default/stream/log_x86_64-555.42.02_faults-new_15_stream/stream_klog"
#log="../../benchmarks/default/stream/log_x86_64-555.42.02_faults-new_12_stream/stream_klog"

ulimit -v 230000000
output_file=`python3 fault_parsing.py $log`
#if [ ! -e $output_file ]; then
  python3 fault_plot.py $log
#else
#    echo "skipping $output_file because it already exists."
#fi


