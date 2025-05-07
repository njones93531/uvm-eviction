#!/bin/bash

./genfaults-subset.sh       #Estimated  3 hours
./evaluate-perf-subset.sh   #Estimated  6 hours
./process-faults-subset.sh  #Estimated <1 hours
./process-perf-subset.sh    #Estimated <1 hours
