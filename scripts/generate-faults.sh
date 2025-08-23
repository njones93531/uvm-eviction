#!/bin/bash
cd ../benchmarks/scriptsv3/

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
[[ -n "$SUBSET" ]] && ARGS="-s"

python3 -u faults.py $ARGS


