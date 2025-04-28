#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
import math

cublasX = [[],[],[]]
cublas_addrs = [[],[],[]]
oversubX = [[],[],[]]
oversub_addrs = [[],[],[]]
o_ymax = [0,0,0]
c_ymax = [0,0,0]
imax = 9999999999999999999999
o_ymin = [imax, imax, imax]
c_ymin = [imax, imax, imax]
c_lines = [[],[],[]]
o_lines = [[],[],[]]

def addrToInt(addr):
    return int(addr, 16)

def roundNice(num):
    digits = math.log10(int(num))
    digits = math.floor(digits)
    factor = math.pow(10, digits)
    return round(float(num) / float(factor)) * factor

def parseFile(filepath, addrs, ymin, i, drawLines):
    with open(filepath, "r") as mister_file:
        print("Opening File")
        csv = mister_file.readlines()
        throwawayVrange = 8713666560
        ymax = 0
        for line in csv:
            garbage = line.split(';')
            cols = garbage[1].split(',')
            if cols[0] == "vrange":
                #store a line in the line list
                # format is vrange, start_addr, size
                a = addrToInt(cols[1])
                if a != throwawayVrange:
                    drawLines.append(a) #start addr
                    print("Line at", cols[1], " -> ", a)
                    if a < ymin[i]:
                        ymin[i] = a
            elif cols[0] == "virt":
                a = addrToInt(cols[1])
                addrs.append(a)
                if a < ymin[i]:
                    ymin[i] = a
                if a > ymax:
                    ymax = a 
                    #print("Ymax at", cols[1], " -> ", a)

def reformAddrs(addrs_list, ymin, ymax, drawLines): 
    #Set up the addrs to be relative integers and set new max of list
    for i in range(len(addrs_list)):
        print("ymin")
        print(ymin[i])
        ymax[i] = 0
        for j in range(len(addrs_list[i])):
            addrs_list[i][j] = addrs_list[i][j] - ymin[i]
            ymax[i] = max(ymax[i], addrs_list[i][j])
        for j in range(len(drawLines[i])):
            drawLines[i][j] = drawLines[i][j] - ymin[i]
            #ymax[i] = max(ymax[i], drawLines[i][j])
        print("Ymax: ", ymax[i]+ymin[i], " -> ", ymax[i])
def determineX(addrs_list, xlist):
    #find numEntries
    numEntries = len(addrs_list)
    #determine xs
    for i in range(0,numEntries):
        xlist.append(i)
    print("xlist")
    print(len(xlist))

#make a generic plot 
def my_plotter(ax, data1, ymin, ymax, drawLines):
    x = []
    determineX(data1, x)
    entries = len(x)
    print(entries)
    print("Entering function my_plotter")  
    ax.plot(x, data1, "b,")
    xm_locator = roundNice(entries/10)
    ym_locator = roundNice(ymax/10)
    print(xm_locator)
    print(entries)
    print(ym_locator)
    print(ymax)
    ax.xaxis.set_major_locator(tkr.MultipleLocator(xm_locator))
    ax.xaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(xm_locator/10))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(ym_locator))
    ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(tkr.MultipleLocator(ym_locator/4))
    ax.set_xlim(0, entries)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Step Number")
    ax.set_ylabel("Memory Location Accessed")
    #ax.grid() #commented out until issue is fixed
    for i in drawLines:
        ax.axhline(i, 0, 1)
        print("Printing line at")
        print(i)
    print("Exiting function my_plotter")

print(len(cublasX))
print("Setting Figure Params")
plt.rcParams['agg.path.chunksize'] = 100000 #I think this can help prevent an overflow error
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(15,15))
#plt.xticks(cublasX) //If you include this, you get OOM error 

#parse cublas files
parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_cublas_1/x86_64-460.27.04_ac-tracking-full_mimc_momc_cublas_1_klog", cublas_addrs[0], c_ymin, 0, c_lines[0])
parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_2m_cublas2/x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_2m_cublas2_klog", cublas_addrs[1], c_ymin, 1, c_lines[1])
parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_64kb_cublas3/x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_64kb_cublas3_klog", cublas_addrs[2], c_ymin, 2, c_lines[2])
reformAddrs(cublas_addrs, c_ymin, c_ymax, c_lines)

#parse oversub files
#parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_oversub/x86_64-460.27.04_ac-tracking-full_mimc_momc_oversub_klog", oversub_addrs[0], o_ymin, 0, o_lines[0])
#parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_2m_oversub2/x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_2m_oversub2_klog", oversub_addrs[1], o_ymin, 1, o_lines[1])
#parseFile("data/log_x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_64kb_oversub3/x86_64-460.27.04_ac-tracking-full_mimc_momc_gran_thold_64kb_oversub3_klog", oversub_addrs[2], o_ymin, 2, o_lines[2])
#reformAddrs(oversub_addrs, o_ymin, o_ymax, o_lines)

print("Attempting Cublas Ax")
#ax1 - Cublas1 
ax1.set_title("Cublas1 Memory Accesses")
my_plotter(ax1, cublas_addrs[0], c_ymin[0], c_ymax[0], c_lines[0])

print("Attempting Cublas Ax")
#ax2 - Cublas2
ax2.set_title("Cublas2 Memory Accesses")
my_plotter(ax2, cublas_addrs[1], c_ymin[1], c_ymax[1], c_lines[1])

print("Attempting Cublas Ax")
#ax3 - Cublas3
ax3.set_title("Cublas3 Memory Accesses")
my_plotter(ax3, cublas_addrs[2], c_ymin[2], c_ymax[2], c_lines[2])

print("Attempting Oversub Ax")
#ax4 - Oversub1
#ax4.set_title("Oversub1 Memory Accesses")
#my_plotter(ax4, oversub_addrs[0], o_ymin[0], o_ymax[0], o_lines[0])

print("Attempting Oversub Ax")
#ax5 - Oversub2
#ax5.set_title("Oversub2 Memory Accesses")
#my_plotter(ax5, oversub_addrs[1], o_ymin[0], o_ymax[1], o_lines[1])

print("Attempting Oversub Ax")
#ax6 - Oversub3
#ax6.set_title("Oversub3 Memory Accesses")
#my_plotter(ax6, oversub_addrs[2], o_ymin[0], o_ymax[2], o_lines[2])

plt.tight_layout()
plt.savefig("figure.png")
plt.close(fig)
