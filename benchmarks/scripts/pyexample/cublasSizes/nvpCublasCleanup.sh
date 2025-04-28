#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J nvp_clean
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1

rm -r ../../default/scripts/pyexample/cublasSizes/tmp/

