#!/bin/bash
#SBATCH -N 1
#SBATCH -w voltron
#SBATCH -J faults
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o slurm_out/%x.%j.out

export IGNORE_CC_MISMATCH=1
module load cuda/12.5
module load mpi

BENCHMARK_DIR="../default/spmm-csr"
BENCHMARK_EXE="./spmm_csr"

if [ -z "$1" ]; then
    echo "Usage: $0 <mode>"
    echo "  mode: normal | managed"
    exit 1
fi

MODE="$1"

if [ "$MODE" != "normal" ] && [ "$MODE" != "managed" ]; then
    echo "Error: mode must be 'normal' or 'managed'"
    exit 1
fi

if [ "$MODE" == "managed" ]; then
	echo "Doing conversion"

	MANAGED_DIR="${BENCHMARK_DIR}_managed"

	echo "Copying benchmark dir"
	rm -rf "$MANAGED_DIR"
	cp -r "$BENCHMARK_DIR" "$MANAGED_DIR"

	echo "Running auto_conv"

	#cudaMalloc -> cudaMallocManaged
	# 	destructive
	input_file="$MANAGED_DIR/spmm_csr_explicit.c"
	output_file="$MANAGED_DIR/spmm_csr_explicit.c"

	bash auto_conv.sh $input_file $output_file

	echo "Building generated benchmark"
	cd "$MANAGED_DIR"

	if ! make -j; then
		echo "Make failed; Conversion requires manual effort"
		scancel $SLURM_JOB_ID
		exit 1
	fi

	## NOTE:
	# 	explicit is used here because ./spmm_csr is already managed
	#	explicit is being used as a test bed for this conversion path
	#	therefore in the future we would like it to be the BENCMARK_EXE
	#		above as the input/output_file and the BENCHMARK_EXE
	#		would hopefully not have to change
	BENCHMARK_DIR="$MANAGED_DIR"
	BENCHMARK_EXE="./spmm_csr_explicit"

	cd - > /dev/null
fi

#export TIMEOUT=12000

python3 -u auto_logs.py \
	--benchmark-dir "$BENCHMARK_DIR" \
	--benchmark-exe "$BENCHMARK_EXE"

cd ~/uvm-eviction/tools/fault_plotsv3

# TODO get rid of hardcoded nonsense
# 	may make more sense doing in auto_logs

# Have already set up env on voltron
#	python3 -m venv ~/venvs/uvm
#	source ~/venvs/uvm/bin/activate
#	pip install --upgrade pip
#	pip install pandas matplotlib numpy psutil

source ~/venvs/uvm/bin/activate
python3 metric_plot.py ../../benchmarks/default/spmm-csr/x86_64-555.42.02_faults-new_1_spmm-csr/klog_spmm-csr.txt -o metrics_stats_relative

# at least set module to default config; should probably reinstall vanilla
# 	done in auto_logs.py
#sudo rmmod nvidia-uvm
#sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
