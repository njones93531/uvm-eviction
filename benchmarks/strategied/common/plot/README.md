##This directory contains scripts that visualize the data from /benchmarks/strategied

#Bash scripts:
- mempress_plotall.sh
  - Generates a figure showing the performance of each allocation pinned to host against migration as problem size grows. Generates figure for each application
- oversub_plotall.sh
  - Geneerates a figure showing the performance of every possible combination of strategies at the 15GB problem size. Generates a figure for each application 
- strat_cluster_all.sh
  - Generates a figure that compares strategies for two different applications

#Python Scripts: 
- perf_data2csv.py 
  - reads the output dump from experiments and parses the relevant statistics into a .csv file. 
  - Usage: python3 perf_data2csv.py <infile>
- oversub_plot.py 
  - Generates a figure showing the performance of every possible combination of strategies at the 15GB problem size. Reads output of perf_data2csv.py
  - Usage: python3 oversub_plot.py <infile>
- mempress_plot.py 
  - Generates a figure shwoing the performance of each allocation pinned to host against migration as problem size grows. Reads output of perf_data2csv.py
  - Usage: python3 mempress_plot.py <infile>
- strategy_cluster.py 
  - Generates a comparison figure for strategies of two different applications 
  - Usage: python3 strategy_cluster.py [-h] -f1 FILE1 -s1 STRATS1 [-l1 LABELS1] -p1 PSIZES1 
                                            -f2 FILE2 -s2 STRATS2 [-l2 LABELS2] -p2 PSIZES2
