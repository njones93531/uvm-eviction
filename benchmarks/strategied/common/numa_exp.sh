#!/bin/bash
logbase=$1
cmd=$2
out=$3
data=${logbase}_numa_pref.data

NUMA_COUNT=$(numactl --hardware 2>/dev/null | awk '/available:/ {print $2}')

# fallback if parsing fails
if ! [[ "$NUMA_COUNT" =~ ^[0-9]+$ ]]; then
  NUMA_COUNT=1
fi

MAX_NODE=$((NUMA_COUNT - 1))

# requested node (you were using 1)
REQ_NODE=1

# clamp to valid range
NUMA_NODE=$(( REQ_NODE > MAX_NODE ? MAX_NODE : REQ_NODE ))

use_timeout=1
if [ $use_timeout -eq 1 ]; then
  echo $out >> $data
  timeout "$TIMEOUT" numactl --cpunodebind=0 --preferred=$NUMA_NODE bash -c "$cmd" >> "$data"
  if [ $? -eq 0 ]; then
    echo "Command did not time out within $TIMEOUT seconds\n" >> $data
  else
    echo "Command timed out after $TIMEOUT seconds\n" >> $data
  fi
else
  echo $out >> $data
  numactl --cpunodebind=0 --preferred=$NUMA_NODE bash -c "$cmd" >> "$data"
fi
