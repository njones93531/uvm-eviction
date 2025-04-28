#!/bin/bash -xe

module load cuda
make

# Original Experiment: Binding to all cores and running fdtd2d
#numactl --physcpubind=0-$(($(nproc)-1))  ./fdtd2d.exe 18 0 -p hhm |& tee hhm.log
#numactl --physcpubind=0-$(($(nproc)-1))  ./fdtd2d.exe 18 0 -p hhd |& tee hhd.log

numactl -H
nvidia-smi topo -m

# Original Experiment: Binding to all cores and running fdtd2d
#numactl --physcpubind=0-$(($(nproc)-1))  ./fdtd2d.exe 18 0 -p hhm |& tee hhm.log
#numactl --physcpubind=0-$(($(nproc)-1))  ./fdtd2d.exe 18 0 -p hhd |& tee hhd.log

# Experiment 1: Binding to one core (core number 8)
echo "Experiment 1: Binding to core 8"
numactl --physcpubind=8 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_core8.log
numactl --physcpubind=8 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_core8.log

# Experiment 2: Binding to one core (core number 7)
echo "Experiment 2: Binding to core 7"
numactl --physcpubind=7 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_core7.log
numactl --physcpubind=7 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_core7.log

# Experiment 3: Binding to all cores and preferring NUMA zone 1
echo "Experiment 3: Binding to all cores and preferring NUMA zone 1 (note membind vs preferred for hhm)"
numactl --physcpubind=0-$(($(nproc)-1)) --membind=1 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa1.log
numactl --physcpubind=0-$(($(nproc)-1)) --preferred=1 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa1.log
numactl --physcpubind=0-$(($(nproc)-1)) --preferred=1 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_numa1.log

# Experiment 4: Binding to all cores and preferring NUMA zone 0
echo "Experiment 4: Binding to all cores and preferring NUMA zone 0 (note membind vs preferred for hhm)"
numactl --physcpubind=0-$(($(nproc)-1)) --membind=0 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa0.log
numactl --physcpubind=0-$(($(nproc)-1)) --preferred=0 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa0.log
numactl --physcpubind=0-$(($(nproc)-1)) --preferred=0 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_numa0.log

# Experiment 5: Binding to core 8 and preferring NUMA zone 1
echo "Experiment 5: Binding to core 8 and preferring NUMA zone 1 (no membind here due to oom)"
numactl --physcpubind=8 --preferred=1 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa1_core8.log
numactl --physcpubind=8 --preferred=1 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_numa1_core.log

# Experiment 6: Binding to core 8 and preferring NUMA zone 0
echo "Experiment 6: Binding to core 8 and preferring NUMA zone 0 (note membind vs preferred for hhm)"
numactl --physcpubind=8 --membind=0 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa0_core8.log
numactl --physcpubind=8 --preferred=0 ./fdtd2d.exe 18 0 -p hhm |& tee hhm_numa0_core8.log
numactl --physcpubind=8 --preferred=0 ./fdtd2d.exe 18 0 -p hhd |& tee hhd_numa0_core8.log
