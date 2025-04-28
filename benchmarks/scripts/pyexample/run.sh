#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J stream-test
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

python3 stream.py


cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream64k_batch-256/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream64k_batch-512/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream64k_batch-1024/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream64k_batch-4096/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream2m_batch-256/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream2m_batch-512/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream2m_batch-1024/*_klog ./
cp ../../default/stream/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream2m_batch-4096/*_klog ./

mv ./*_klog ~/uvm-eviction/tools/eviction_calc/stream_klogs











