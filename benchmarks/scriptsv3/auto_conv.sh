#!/bin/bash
# Usage: ./auto_conv.sh input_file.c [output_file.c]

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 input_file.c [output_file.c]"
    exit 1
fi

input_file="$1"
output_file="${2:-$input_file.tmp}"

cp $input_file $output_file

sed -i -E 's/cudaMalloc\(([^,]+),([^)]+)\)/cudaMallocManaged(\1, \2), cudaMemAttachGlobal/g' "$output_file"

sed -i '/cudaMemcpy/d' "$output_file"

sed -i -E 's/\bd([A-Z_][a-zA-Z0-9_]*)\b/h\1/g' "$output_file"
