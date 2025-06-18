#!/bin/bash

TIMEOUT=1200
./genfaults.sh -s -t $TIMEOUT 	#Estimated  3 hours
./evaluate-perf.sh -s -t $TIMEOUT  #Estimated  6 hours
./process-faults.sh -s -t $TIMEOUT #Estimated <1 hours
./process-perf.sh -s -t $TIMEOUT   #Estimated <1 hours
