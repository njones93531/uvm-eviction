#!/bin/bash
logbase=$1
cmd=$2
out=$3
data=${logbase}_numa_pref.data

echo $out >> $data
if timeout $TIMEOUT time 2>> $data numactl --cpunodebind=0 --preferred=1 $cmd >> $data; then
	echo "Command did not time out\n" >> $data
else
	echo "Command timed out after $TIMEOUT seconds\n" >> $data
	exit
fi

