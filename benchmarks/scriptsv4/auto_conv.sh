#!/bin/bash
# Usage: ./auto_conv.sh input_file.c [output_file.c]

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 input_file.c [output_file.c]"
    exit 1
fi

input_file="$1"
output_file="${2:-$input_file.tmp}"

cp $input_file $output_file

# Remove h* pointer declarations
# 	right now we are avoiding hC_results because we have host side correctness checks
# sed -i -E '/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*h[A-Za-z0-9_]+\s*;/d' "$output_file"
sed -i -E '/^\s*float\*\s*hC_result\s*;/! s/^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\*\s*h[A-Za-z0-9_]+\s*;//' "$output_file"

# Remove h* malloc lines
# 	right now we are avoiding hC_results because we have host side correctness checks
# sed -i -E '/^\s*h[A-Za-z0-9_]+\s*=\s*\([^)]*\)\s*malloc\s*\([^;]*\)\s*;/d' "$output_file"
sed -i -E '/^\s*hC_result\s*=\s*\([^)]*\)\s*malloc\s*\([^;]*\)\s*;/! { 
    /^\s*h[A-Za-z0-9_]+\s*=\s*\([^)]*\)\s*malloc\s*\([^;]*\)\s*;/d
}' "$output_file"

# Edge case for mallocs with no sizeof
sed -i -E '/cudaMalloc\(/ {
    /sizeof/ ! s/cudaMalloc\s*\(\s*(&[a-zA-Z0-9_]+)\s*,\s*([^)]+)\)/cudaMallocManaged(\1, \2, cudaMemAttachGlobal)/g
}' "$output_file"

# cudaMalloc -> cudaMallocManaged
sed -i -E 's/cudaMalloc\(([^,]+),([^)]+)\)/cudaMallocManaged(\1, \2), cudaMemAttachGlobal/g' "$output_file"

# Remove cudaMemcpys
sed -i '/cudaMemcpy/d' "$output_file"

# Rename d* variables -> h*
sed -i -E 's/\bd([A-Z_][a-zA-Z0-9_]*)\b/h\1/g' "$output_file"

# Replace all free(h*) with cudaFree(h*)
sed -i -E 's/^\s*free\s*\(\s*(h[A-Za-z0-9_]*)\s*\)\s*;/cudaFree(\1);/g' "$output_file"
