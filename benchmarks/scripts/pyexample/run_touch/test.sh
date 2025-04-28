#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J ac-tracking
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-1/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-2/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-3/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-4/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-5/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-1/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-2/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-3/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-4/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-5/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch64k-8t/*_klog ./
cp ../../../demo/access_counters/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_touch2m-8t/*_klog ./
mv *_klog ../../../tools/eviction_calc/touch_klogs/

cd ../../../tools/eviction_calc/
sbatch run.sh
