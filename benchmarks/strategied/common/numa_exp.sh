#!/bin/bash
logbase=$1
cmd=$2
out=$3
data=${logbase}_numa_pref.data


use_timeout=1
if [ $use_timeout -eq 1 ]; then
  echo $out >> $data
  if timeout $TIMEOUT $(numactl --cpunodebind=0 --preferred=1 $cmd >> $data); then
    echo "Command did not time out within $TIMEOUT seconds\n" >> $data
  else
    echo "Command timed out after $TIMEOUT seconds\n" >> $data
  fi
else
  echo $out >> $data
  numactl --cpunodebind=0 --preferred=1 $cmd >> $data
fi

