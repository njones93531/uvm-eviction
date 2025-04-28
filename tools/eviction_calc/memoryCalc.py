#!/usr/bin/python3
import sys
import heapq
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
import math 




BLOCK_SIZES = [] #NUMBER OF BYTES IN ONE BLOCK OF MEMORY
DATA_FILES = [] #FORMATTED .CSV FILES FROM WHICH THE ACCESS DATA IS PULLED
TITLES = [] #TITLES OF EACH GRAPH YOU WANT IN THE OUTPUT
addrs = []
accesses_dicts = []
MEM_SIZE = 0 #NUMBER OF BYTES THAT FIT IN MEMORY



if(len(sys.argv) < 2):
    print("Usage: python3 memoryCalc.py <filepath for config> [opt: hline height1] [opt: hline height2]...")
    sys.exit("Number of arguments:", len(sys.argv))



      
def addrToIndex(addr, i):
    return int(addr, 16)/BLOCK_SIZES[i]

def parseFile(it): 
    accesses_dict = {}
    addrs = []
    with open(DATA_FILES[it], "r") as mister_file:
        print("Opening File")
        csv = mister_file.readlines()
        step = 0
        for line in csv:
            line = addrToIndex(line, it)
            block_index = int(line)
            if (line not in accesses_dict):
                accesses_dict[block_index]=[0] #The first element is the pointer to current position
            accesses_dict[block_index].append(step)
            step = step + 1
            addrs.append(block_index)
    return accesses_dict, addrs

       


def writeMem(evict_fp, reside_fp, accesses_dict, addrs, it): #todo: fix this function to a) do the plotting inside this file and b) work with arrays of files
    h = []
    with open(evict_fp, "w") as evict_file:
        step = 0
        for addr in addrs:
            block_index = int(addr)
            if(len(h) == MEM_SIZE):
                if(h[0][1] != block_index):
                    evict_file.write(str(step) + ', ' + str(h[0][1]) + '\n')
                heapq.heappop(h)
            heapq.heappush(h, (accesses_dict[block_index][accesses_dict[block_index][0]], block_index))
            accesses_dict[block_index][accesses_dict[block_index][0]] = accesses_dict[block_index][accesses_dict[block_index][0]] + 1
            step = step + 1

def roundNice(num):
    digits = math.log10(int(num))
    digits = math.floor(digits)
    factor = math.pow(10, digits)
    return round(float(num) / float(factor)) * factor

def getAddrExtremes(addrs):
    ymax = 0
    ymin = 9999999999999999999999999
    for i in addrs:
        ymax = max(i, ymax)
        ymin = min(i, ymin)
    return ymin, ymax

def getMaxAccess(accesses_dict):
    maxAccess = 0
    for i in accesses_dict:
        maxAccess = max(maxAccess, len(accesses_dict[i])) 
    return maxAccess

def adjAddr(addr, ymin):
    return addr-ymin

def subplotWrapper():
    fig, ax = plt.subplots(len(BLOCK_SIZES), 1, figsize = (15, 8 * len(BLOCK_SIZES))) 
    if (len(BLOCK_SIZES)==1):
        axList = []
        axList.append(ax)
        return fig, axList
    return fig, ax

def ticks(ax, ymin, ymax, entries, xLowBound, yLowBound):
     #Tickmark and Axis Formatting
    #xm_locator = roundNice(entries/10)
    #ym_locator = roundNice(ymax/10)
    #ax.xaxis.set_major_locator(tkr.MultipleLocator(xm_locator))
   # ax.xaxis.set_major_formatter(tkr.StrMethodFormatter('{x:g}'))
   # ax.xaxis.set_minor_locator(tkr.MultipleLocator(xm_locator/10))
   # ax.yaxis.set_major_locator(tkr.MultipleLocator(ym_locator))
   # ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
   # ax.yaxis.set_minor_locator(tkr.MultipleLocator(ym_locator/4))
   # ax.set_ylim(yLowBound, ymax)
   # ax.set_xlim(xLowBound, entries)
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x',useMathText=True)

def evictionPlot(ax, fig, step, evicts):
    ymin, ymax = getAddrExtremes(evicts) 
    entries = len(step)
    ymax = adjAddr(ymax, ymin)
    evicts = [x-ymin for x in evicts] #adjust each address by mapping the min to 0

    ax.set_xlabel("Memory Access Event")
    ax.set_ylabel("Memory Block Evicted")
    ticks(ax, 0, ymax, entries, step[0], 0)
    if(len(sys.argv) > 2):
        for i, height in enumerate(sys.argv):
            if(i > 1):
                ax.axhline(y=int(height), color='b')


    print("Entering the call to scatter")
    print("Size of step: " + str(len(step)))
    print("Size of evicts: " + str(len(evicts)))
    sc = ax.scatter(step, evicts, s=0.1, linewidths=0)
    return sc 

def gradientPlot(ax, fig, addrs, addrs_dict):
    ymin, ymax = getAddrExtremes(addrs) 
    entries = len(addrs)
    ymax = adjAddr(ymax, ymin)

    ticks(ax, ymin, ymax, entries, 0, 0)
    ax.set_xlabel("Memory Access Event")
    ax.set_ylabel("Memory Block Accessed")
    ax.set_facecolor("#D3D3D3")
    if(len(sys.argv) > 2):
        for i, height in enumerate(sys.argv):
            if(i > 1):
                ax.axhline(y=int(height), color='b')

    #Plotting using Scatter and Color List
    maxAccess = getMaxAccess(addrs_dict)
    colors = []
    step = 0
    x = []
    y = []
    for i in addrs:
        #create the gradient
        d = addrs_dict[i][0]
        colors.append(int(d))

        x.append(step)
        step = step + 1
        y.append(adjAddr(i, ymin))
        addrs_dict[i][0] = addrs_dict[i][0] + 1
    print("Entering the call to scatter")
    sc = ax.scatter(x, y, c=colors, cmap='inferno_r', s=0.1, linewidths=0)
    return sc

def eviction():
    fig, ax = subplotWrapper()
    if(len(ax) > 1):
        ax.flatten()
    for it in range(0, len(BLOCK_SIZES)):
        accesses_dict, addrs = parseFile(it)
        h = []
        step = 0
        steps = []
        evicts = []
        for addr in addrs:
            block_index = int(addr)
            if(len(h) == MEM_SIZE):
                if(h[0][1] != block_index):
                    steps.append(step)
                    evicts.append(h[0][1])
                heapq.heappop(h)
            heapq.heappush(h, (accesses_dict[block_index][accesses_dict[block_index][0]], block_index))
            accesses_dict[block_index][accesses_dict[block_index][0]] = accesses_dict[block_index][accesses_dict[block_index][0]] + 1
            step = step + 1
        print("size of steps: " + str(len(steps)))
        evictionPlot(ax[it], fig, steps, evicts)
        ax[it].set_title(TITLES[it])
    plt.tight_layout()
    plt.savefig("evictions.png")
    plt.close(fig)

def unique():
    for it in range(0, len(BLOCK_SIZES)):
        access_dict, addr = parseFile(it)
        print("File " + str(it) + " has " + str(len(access_dict)) + " unique addresses ")


def gradient():        #TODO args are unnecessary
    fig, ax = subplotWrapper()
    if(len(ax) > 1):
        ax.flatten()
    for it in range(0,len(BLOCK_SIZES)):
        access_dict, addr = parseFile(it)
        sc = gradientPlot(ax[it], fig, addr, access_dict)
        cbar = fig.colorbar(sc, ax=ax[it], label='Access Count')
        ax[it].set_title(TITLES[it])
        #cbar.formatter.set_powerlimits((0, 0))
        #cbar.formatter.set_useMathText(True)
    plt.tight_layout()
    plt.savefig("gradient.png")
    plt.close(fig)

print("Number of arguments:", len(sys.argv))
with open(sys.argv[1], "r") as config:
    #reading config
    print("Reading Config")
    lines = config.readlines()
    mode = lines[0].strip().split(': ')[1]
    BLOCK_SIZES = list(map(int, lines[1].strip().split(': ')[1].split(', ')))
    DATA_FILES = lines[2].strip().split(': ')[1].split(', ')
    TITLES = lines[3].strip().split(': ')[1].split(', ')
    if(len(BLOCK_SIZES)!=len(DATA_FILES) or len(BLOCK_SIZES)!=len(TITLES)):
            sys.exit("Invalid config: mismatching list sizes")
    MEM_SIZE = lines[4].strip().split(': ')[1]

    if(mode == "gradient"):
        #initialization for gradient things
        gradient()                

    elif(mode == "eviction"):
        eviction()

    elif(mode == "unique"):
        unique()
        

        #if(sys.argv[1] == "print"):
         #   evict_fp = "./evictions.out"
          #  writeMem(evict_fp, "null", accesses_dict, addrs)
    else:
        print(mode)















