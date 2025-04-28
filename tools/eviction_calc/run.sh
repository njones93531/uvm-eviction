#!/bin/bash
#SBATCH -N 1
#SBATCH -w n01
#SBATCH -J accessPlot
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o %x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda

python3 addressGrabber.py
#python3 tempRTGrabber.py
python3 memoryCalc.py mcConfigGrad.txt
#python3 memoryCalcTemp.py mcConfigTouch.txt
