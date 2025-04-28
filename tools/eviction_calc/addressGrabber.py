#!/usr/bin/python3
import numpy as np
import math

filesToParse = []
for mem in [25, 50]:
    for comp in [25, 50, 75, 100]:
        filesToParse.append(f"/home/najones/uvm-eviction/benchmarks/UVMBench/bfs/log_x86_64-535.104.05_ac-tracking-full_mimc_momc_gran_nopf_thold_uvmbench_bfs_{mem}gpumem_{comp}complete/x86_64-535.104.05_ac-tracking-full_mimc_momc_gran_nopf_thold_uvmbench_bfs_{mem}gpumem_{comp}complete_klog")

#for x in range(8, 17):
 #   b.append(x)
#for t in b:
 #   for gran in {"2m", "64k"}:
  #      for i in range(0,1):
   #         filesToParse.append("stream_klogs/x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold__stream" + gran + "_batch-" + str(t) + "_klog")
wc =0

def parseFile(filepath):
    wc = 0
    with open(filepath, "r") as mister_file:
        print("Opening File")
        with open(filepath[:-5]+"_formatted.csv", "w") as write_file:
            csv = mister_file.readlines()
            for line in csv:
                garbage = line.split(';')
                cols = garbage[1].split(',')
                #if cols[0] == "vrange":
                    #we dont care about this yet, will send to a diff file
                if cols[0] == "virt":
                    if cols[1][-1] != '\n':
                        cols[1] = cols[1] + '\n'
                    write_file.write(cols[1])
                    wc = wc +1
    print(i + "          wc: " + str(wc))

c = 0
for i in filesToParse:
    parseFile(i)
    c = c+1
    if c%5 == 0:
        print()
