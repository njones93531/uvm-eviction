#!/bin/bash

TIMEOUT=12000
SUBSET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) TIMEOUT="$2"; shift 2 ;;
    -s|--subset) SUBSET=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -n "$TIMEOUT" ]] && export TIMEOUT=$TIMEOUT
[[ -n "$SUBSET" ]] && ARGS="$ARGS -s" 

cd ../benchmarks/strategied/common/plot/
./csv_all.sh
cd - 

cd ../tools/fault_plotsv3/
python3 parse_metrics_relative.py $ARGS
[[ -z "$SUBSET" ]] && python3 -u mvt_example.py


