#!/bin/bash

mig=mmmmmmmmmmmmmmmmmmmm

#for size in 12 15 18 21
#  do
#  strat=$mig
#  echo "$size $strat" >> exp.out
#  cp tea$size.in tea.in
#  numactl --physcpubind=8-15 --preferred=1 ./tealeaf -p $strat | grep "Total elapsed" >> exp.out
#done

size=12
strat=ddddddddddddddmmmmmm
strat=dddddmddddddddmmmmmm
echo "$size $strat" >> exp.out
cp tea$size.in tea.in
numactl --physcpubind=8-15 --preferred=1 ./tealeaf -p $strat | grep "Total elapsed" >> exp.out
size=15
strat=ddddddddddddddmmmmmm
strat=ddddmdmmdmdddmmmmmmm
echo "$size $strat" >> exp.out
cp tea$size.in tea.in
numactl --physcpubind=8-15 --preferred=1 ./tealeaf -p $strat | grep "Total elapsed" >> exp.out
size=18
strat=dddddmdmddddddmmmmmm
strat=dddmmmmmddmmdmmmmmmm
echo "$size $strat" >> exp.out
cp tea$size.in tea.in
numactl --physcpubind=8-15 --preferred=1 ./tealeaf -p $strat | grep "Total elapsed" >> exp.out
size=21
strat=dddmddhddhddmdmmmmmm
strat=dmdmdmmmdmdmmmmmmmmm
echo "$size $strat" >> exp.out
cp tea$size.in tea.in
numactl --physcpubind=8-15 --preferred=1 ./tealeaf -p $strat | grep "Total elapsed" >> exp.out

