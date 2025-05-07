#!/bin/bash -xe

cd ../benchmarks/scriptsv3/

export TIMEOUT=1200
python3 -u faults.py --subset

