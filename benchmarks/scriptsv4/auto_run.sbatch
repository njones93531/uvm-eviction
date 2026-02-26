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

KERNEL_VERSION="x86_64-555.42.02"
KERNEL_VARIANT="faults-new"

RELATIVE_FIG_DIR="../../figs/voltron/metrics_stats_relative"

if [ -z "$1" ]; then
    echo "Usage: $0 <mode> <dir>"
    echo "  mode: normal | managed"
    exit 1
fi

MODE="$1"

#given something like:
#	../default/spmm-csr
BENCHMARK_DIR="$2"
#	spmm-csr
BENCH_NAME=$(basename "$BENCHMARK_DIR")

#TODO figure some way to find which file to edit and 
# 	what the exe is named
BENCHMARK_EXE="./spmm_csr"

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

#TODO needs to take benchmark psize as a variable
#	then we can use that later in this bash
python3 -u auto_logs.py \
	--benchmark-dir "$BENCHMARK_DIR" \
	--benchmark-exe "$BENCHMARK_EXE" \
	--args 1 0.1 1

cd ~/uvm-eviction/tools/fault_plotsv4

# Have already set up env on voltron
#	python3 -m venv ~/venvs/uvm
#	source ~/venvs/uvm/bin/activate
#	pip install --upgrade pip
#	pip install pandas matplotlib numpy psutil
source ~/venvs/uvm/bin/activate

# TODO get rid of hardcoded nonsense
# 	may make more sense doing in auto_logs
#		size?
#		
output=$(python3 main.py $RELATIVE_FIG_DIR/$BENCH_NAME/default_${KERNEL_VERSION}_${KERNEL_VARIANT}_1_$BENCH_NAME.txt)
strategy=$(echo "$output" | grep "Final Strategy:" | sed 's/Final Strategy: //')
echo $strategy

#TODO apply strategy and run it.
#	this is still horribly hardcoded
cd ~/uvm-eviction/benchmarks/scriptsv4

STRATEGY_DIR="../strategied/${BENCH_NAME}_strat"

echo "Copying benchmark dir"
rm -rf "$STRATEGY_DIR"
cp -r "$BENCHMARK_DIR" "$STRATEGY_DIR"

if [ "$MODE" == "managed" ]; then
	IN_FILE="$STRATEGY_DIR/spmm_csr_explicit.c"
	OUT_FILE="${IN_FILE}_new.c"
else
	IN_FILE="$STRATEGY_DIR/spmm_csr.c"
	OUT_FILE="${IN_FILE}_new.c"
fi

bash apply_strat.sh $IN_FILE $strategy >  $OUT_FILE
mv $OUT_FILE $IN_FILE

echo "Building generated benchmark"
cd "$STRATEGY_DIR"

if ! make -j; then
	echo "Make failed; Conversion requires manual effort"
	scancel $SLURM_JOB_ID
	exit 1
fi

$BENCHMARK_EXE 1 0.1 1

# at least set module to default config; should probably reinstall vanilla
# 	done in auto_logs.py
#sudo rmmod nvidia-uvm
#sudo modprobe nvidia-uvm #uvm_perf_prefetch_enable=0 uvm_perf_fault_coalesce=0 #uvm_perf_fault_batch_count=1
