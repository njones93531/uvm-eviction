#!/bin/bash -xe

date=$(date +"%Y-%m-%d-%H-%M-%S")
KB_per_GB=1000000
total_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
total_gb=$((total_kb / $KB_per_GB))
running_jobs=0
memory_used=0

resubmit_prefix=""

for size_GB in $((23 * $total_gb / 100)) $((45 * $total_gb / 100)) $((95 * $total_gb / 100)); do
  size_KB=$((size_GB * KB_per_GB))

  for log in $(find ../../benchmarks/default/ -name '*_klog'); do
    found=false
    for item in "${blacklist[@]}"; do
      if [ "$item" == "$log" ]; then
        found=true
        break
      fi
    done

    if [ "$found" != true ]; then
      job_name="${resubmit_prefix}plot_${date}_$(basename $(dirname "$log"))"
      log_dir="slurm_out"
      mkdir -p "$log_dir"
      slurm_file="${log_dir}/${job_name}.out"

      output_file=$(python3 fault_parsing.py "$log" metrics_stats_relative)
      if [ ! -e "$output_file" ]; then
        (
          ulimit -v $size_KB
          python3 metric_plot.py "$log" -o metrics_stats_relative
        ) &> "$slurm_file" &

        ((running_jobs+=1))
        ((memory_used+=$size_GB))
        while [ "$(($memory_used + $size_GB))" -ge "$total_gb" ]; do
          wait -n  # wait for any job to finish before launching a new one
          ((running_jobs-=1))
          ((memory_used-=$size_GB))
        done
      else
        echo "skipping $output_file because it already exists."
      fi
    else
      echo "skipping $log because it is on the blacklist."
    fi
  done

  # Wait for all background jobs from this batch to complete
  wait
  resubmit_prefix="resubmit_"
done

