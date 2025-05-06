#!/bin/bash -xe

cd ../benchmarks/scriptsv3/

export TIMEOUT=120
python3 -u faults.py --subset

