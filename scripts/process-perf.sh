#!/bin/bash

cd ../benchmarks/strategied/common/plot/
./csv_all.sh
cd - 

cd ../tools/fault_plotsv3/
python3 parse_metrics_relative.py
python3 mvt_example.py
