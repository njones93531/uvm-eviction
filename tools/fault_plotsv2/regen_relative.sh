#!/bin/bash -xe

# Ensure the slurm_out directory exists
mkdir -p slurm_out

# Array to store jobs to check later
declare -a job_ids

blacklist=() #"../../benchmarks/tyler-default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_21_GRAMSCHM/GRAMSCHM_klog", "../../benchmarks/tyler-default/GRAMSCHM/log_x86_64-555.42.02_faults-new_nopf_18_GRAMSCHM/GRAMSCHM_klog")
KB_per_GB=1000000
resubmit_prefix=""
for size_GB in 60 120 240; do
  size_KB=$((size_GB * KB_per_GB))
  for log in $(find ../../benchmarks/tyler-default-simple/ -name '*_klog'); do
      found=false
      for item in "${blacklist[@]}"; do
          if [ "$item" == "$log" ]; then
              found=true
              break
          fi
      done
      if [ "$found" != true ]; then
   
        # Prepare the filename for the SLURM output
        job_name="${resubmit_prefix}plot_$(basename $(dirname $log))"
        slurm_file="slurm_out/${job_name}.out"

        output_file=`python3 fault_parsing.py $log`
        if [ ! -e $output_file ]; then
            # Submitting the job to SLURM with the correct output file and partition
            job_id=$(sbatch --parsable --job-name=$job_name --output=$slurm_file --partition=hsw --time=4:00:00 --mem=${size_GB}G --cpus-per-task=4 --wrap="ulimit -v ${size_KB}; python3 fault_plot.py $log")
            job_ids+=($job_id)
        else
            echo "skipping $output_file because it already exists."
        fi
      else
        echo "skipping $item because it is on the blacklist."
      fi
  done

# Wait for all jobs to complete
  all_done=0
  while [ $all_done -eq 0 ]; do
      all_done=1
      for job_id in "${job_ids[@]}"; do
          status=$(sacct -j $job_id --format=State --noheader | head -n 1 | awk '{print $1}')
          if [[ $status == "RUNNING" ]] || [[ $status == "PENDING" ]]; then
              all_done=0
              break
          fi
      done
      sleep 10  # Check every 10 seconds
  done
  resubmit_prefix=resubmit_
done



## Check job status and re-submit failed jobs with exclusive resources
#for job_id in "${job_ids[@]}"; do
#    status=$(sacct -j $job_id --format=State --noheader | head -n 1 | awk '{print $1}')
#    if [[ $status != "COMPLETED" ]]; then
#        log=$(sacct -j $job_id --format=JobName%256 --noheader | sed 's/plot_//g' | awk '{print "../../benchmarks/tyler-default/" $1 "_klog"}')
#        job_name="retry_plot_$(basename $log)"
#        slurm_file="slurm_out/${job_name}.out"
#
#        # Resubmit the failed job with exclusive resources
#        sbatch --job-name=$job_name --output=$slurm_file --partition=hsw --time=1:00:00 --exclusive --wrap="ulimit -v 230000000; python3 metric_plot.py $log"
#    fi
#done
#
