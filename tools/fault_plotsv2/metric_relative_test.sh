#!/bin/bash -xe

# Ensure the slurm_out directory exists
mkdir -p slurm_out

# Array to store jobs to check later
declare -a job_ids

log="../../benchmarks/tyler-default/FDTD-2D/log_x86_64-555.42.02_faults-new_15_FDTD-2D/FDTD-2D_klog"
# Prepare the filename for the SLURM output
job_name="plot_metric_$(basename $(dirname $log))"
slurm_file="slurm_out/${job_name}.out"

output_file=`python3 fault_parsing.py $log metrics`
if [ ! -e $output_file ]; then
    # Submitting the job to SLURM with the correct output file and partition
    job_id=$(sbatch --parsable --job-name=$job_name --output=$slurm_file --partition=hsw --time=1:00:00 --mem=40G --cpus-per-task=4 --wrap="python3 metric_plot.py $log")
    job_ids+=($job_id)
else
    echo "skipping $output_file because it already exists."
fi

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


