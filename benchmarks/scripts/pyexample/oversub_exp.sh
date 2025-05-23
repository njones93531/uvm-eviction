#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J oversub_exp
#SBATCH --exclusive
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

python3 oversubAll.py

cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-16-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-16-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-15-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-15-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-14-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-14-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-13-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-13-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-12-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-12-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-11-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-11-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-10-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-10-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-9-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-9-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-8-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-8-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub2m-1-9/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-0/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-1/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-2/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-3/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-4/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-5/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-6/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-7/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-8/*_klog ./
cp ../../../demo/unified-memory-oversubscription/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_oversub64k-1-9/*_klog ./
mv *_klog ../../../tools/eviction_calc/oversub_klogs/

#cd ../../../tools/eviction_calc/
#sbatch run.sh
