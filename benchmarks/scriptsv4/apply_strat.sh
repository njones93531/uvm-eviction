#!/usr/bin/env bash

set -e

FILE="$1"
POLICY="$2"

if [[ -z "$FILE" || -z "$POLICY" ]]; then
    echo "Usage: $0 <file.c> <policy_string (e.g. hhddm)>"
    exit 1
fi

policy_idx=0

awk -v policy="$POLICY" '
function next_policy() {
    p = substr(policy, idx+1, 1)
    idx++
    return p
}

BEGIN {
    idx = 0
}

/cudaMallocManaged[[:space:]]*\(/ {

    mode = next_policy()

    # Extract pointer name and size
    # Matches: &ptr , size ,
    match($0, /\&[[:space:]]*([a-zA-Z0-9_]+)[[:space:]]*,[[:space:]]*([^,]+),/, m)
    ptr = m[1]
    size = m[2]

    if (mode == "m" || mode == "") {
        print $0
    }

    else if (mode == "d") {
        print "{"
        print "    void *devptr;"
        print "    cudaMalloc(&devptr, " size ");"
	print "    //CHECK_CUDA_ERROR();"
        print "    cudaMemcpy(devptr, " ptr ", " size ", cudaMemcpyHostToDevice);"
	print "    //CHECK_CUDA_ERROR();"
        print "    cudaFree(" ptr ");"
	print "    //CHECK_CUDA_ERROR();"
        print "    " ptr " = devptr;"
        print "}"
    }

    else if (mode == "h") {
        print "cudaMallocManaged((void **) &" ptr ", " size ", cudaMemAttachGlobal);"
        print "cudaMemAdvise(" ptr ", " size ", cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);"
        print "cudaMemAdvise(" ptr ", " size ", cudaMemAdviseSetAccessedBy, 0);"
    }

    else {
        print "// Unknown policy: " mode
        print $0
    }

    next
}

{ print }

' "$FILE"

