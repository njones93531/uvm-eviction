#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J ac-tracking
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

python3 run_touch.py > rt_klog
python3 run_touch2.py > rt2_klog
python3 run_touch3.py > rt3_klog
python3 run_touch4.py > rt4_klog
python3 run_touch5.py > rt5_klog
python3 run_touch6.py > rt6_klog
python3 run_touch7.py > rt7_klog
python3 run_touch8.py > rt8_klog
python3 run_touch9.py > rt9_klog
python3 run_touch10.py > rt10_klog
python3 run_touch11.py > rt11_klog
python3 run_touch12.py > rt12_klog


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
