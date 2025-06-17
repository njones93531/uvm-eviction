#!/bin/bash

cd ../benchmarks/scriptsv3/

export TIMEOUT=12000
python3 -u perf.py -t $TIMEOUT
python3 -u mvt-example.py -t $TIMEOUT

