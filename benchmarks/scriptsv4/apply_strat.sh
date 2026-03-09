#!/usr/bin/env bash

set -e

FILE="$1"
POLICY="$2"

if [[ -z "$FILE" || -z "$POLICY" ]]; then
    echo "Usage: $0 <file.c> <policy_string (e.g. hhddm)>"
    exit 1
fi

policy_idx=0

read -r -d '' CODE << 'EOF' || true
void setAllocationPolicy(void **a, std::size_t size, char flag) {
	switch(flag){
		case 'm': //Policy is migrate; do nothing
			break;
		case 'd': //Pin to device; use cudaMemCopy
			void * devptr;
			cudaMalloc(&devptr, size);
			cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
			CUDA_CHECK(cudaFree(*a));
			*a = devptr;
			break;
		case 'a': //Pin to device, async variant
			//First, set preferred location of everything to device
			cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
			//Finally, move the pin device mem to device
			cudaMemPrefetchAsync(*a, size, 0, 0);
			break;
		case 'h': //Pin to host; use cudaMemAdvise
			cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
			cudaMemAdvise(*a, size, cudaMemAdviseSetAccessedBy, 0);
			break;
		default:
			std::cout << "Policy flag '" << flag << "' used on allocation " << " is not supported.\n";
			exit(1);
	}
	return;
}



EOF

TMPFILE=$(mktemp)
echo "$CODE" | cat - "$FILE" > "$TMPFILE" && mv "$TMPFILE" "$FILE"

# Extract pointer names
mapfile -t POINTERS < <(grep -oP 'cudaMallocManaged\(\(void\*\*\)\s*&\K[a-zA-Z_][a-zA-Z0-9_]*' "$FILE")

# Extract sizes (everything between the second comma and closing paren)
mapfile -t SIZES < <(grep -oP 'cudaMallocManaged\([^,]+,\s*\K[^,]+(?=,\s*cudaMemAttachGlobal)' "$FILE")

if [[ ${#POINTERS[@]} -eq 0 ]]; then
    echo "No cudaMallocManaged calls found in '$FILE'."
    exit 1
fi

echo "Found ${#POINTERS[@]} allocations:"
for i in "${!POINTERS[@]}"; do
    echo "  ${POINTERS[$i]} -> ${SIZES[$i]}"
done

# Applying strat
for i in "${!POINTERS[@]}"; do
    flag="${POLICY:$i:1}"
    if [[ -z "$flag" ]]; then
        break 
    fi
    echo "setAllocationPolicy((void**) &${POINTERS[$i]}, ${SIZES[$i]}, '${flag}');"
done

# Build the function calls as a block - write to temp file instead
CALLS_FILE=$(mktemp)
for i in "${!POINTERS[@]}"; do
    flag="${POLICY:$i:1}"
    if [[ -z "$flag" ]]; then
        break
    fi
    echo "    setAllocationPolicy((void**) &${POINTERS[$i]}, ${SIZES[$i]}, '${flag}');" >> "$CALLS_FILE"
done

TMPFILE=$(mktemp)
while IFS= read -r line; do
    echo "$line"
    if [[ "$line" == *"// PLACE ALLOCATION POLICY"* ]]; then
        cat "$CALLS_FILE"
    fi
done < "$FILE" > "$TMPFILE" && mv "$TMPFILE" "$FILE"
rm "$CALLS_FILE"
