#!/bin/bash -xe

psize=`expr 1024 \* 28`
./cublas-vanilla.sh  $psize |& tee cublas-vanilla.log
./cublas-asyncevict.sh $psize |& tee cublas-asyncevict.log
