#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import math

INFILE = "formatted0-14.txt"
MARKERS = ["o", "v", "8", "s", "p", "P", "*", "h", "H", "X", "D", "^", "<", ">", "1", "2", "3", "4", "+", "x", "d"] #included for accessibility

NUM_COLUMNS = 28
MEMS = [6, 9, 12, 15, 18, 21, 24]
STRIDES = [1, 2, 30, 32, 34, 60, 64, 68, 120, 128, 136, 252, 256, 260, 500] 

def roundNice(num):
    if(num<=0):
        return num
    digits = math.log10(int(num))
    digits = math.floor(digits)
    factor = math.pow(10, digits)
    return round(float(num) / float(factor)) * factor

def parseFile(filepath, x_col, y_col, args):
    with open(filepath, "r") as mister_file:
        print("Opening File")
        csv = mister_file.readlines()
        xmax = 0
        ymax = 0
        x =[]
        y = []
        print(f"Length {len(csv)}")
        for line in csv:
            cols = line.split(',')
            includeLine = True
            for i in range(0, len(args)): 
                if (len(cols)<NUM_COLUMNS):
                    includeLine = False
                if (includeLine and i!=x_col and i!=y_col and str(cols[i])!=str(args[i])):
                    includeLine = False
            if includeLine:
                x.append(int(cols[x_col]))
                y.append(float(cols[y_col][:-2]))
                xmax = max(xmax, int(cols[x_col]))
                ymax = max(ymax, float(cols[y_col][:-2]))
        return x,y,xmax,ymax


#make a generic plot
def my_plotter(ax, x, y, xmax, ymax, xlabel, ylabel, mems):
    print("xmax, ymax " + str(xmax) + " " + str(ymax));

    xmax = int(xmax * 1.2)
    ymax = int(ymax * 1.2)
    labels = []
    colors = []
    loopMax = 0
    if mems: 
        colors = cm.viridis_r(np.linspace(0, 1, len(MEMS)))
        loopMax = len(MEMS)
        for i in range(0, loopMax):
            labels.append(f"{MEMS[i]} GB")
    else:
        colors = cm.viridis_r(np.linspace(0, 1, len(STRIDES)))
        loopMax = len(STRIDES)
        for i in range(0, loopMax):
            labels.append(f"Stride of {STRIDES[i]}")
 
    for i in range(0, loopMax):
        ax.scatter(x[i], y[i], color=colors[i], marker=MARKERS[i], s=150, label=labels[i], alpha=0.7)
    xm_locator = roundNice(xmax/10)
    ym_locator = roundNice(ymax/10)
    ax.xaxis.set_major_locator(tkr.MultipleLocator(xm_locator))
    ax.xaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(xm_locator/4))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(ym_locator))
    ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    ax.yaxis.set_minor_locator(tkr.MultipleLocator(ym_locator/4))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()

def callPlotter(ax, axx, axy, xCol, yCol, title, xlabel, ylabel, read, remote_access):
    ax[axx][axy].set_title(title)
    filepath = INFILE #"formatted.out"
    ymax = 0
    xmax = 0
    realy = []
    realx = []
    if(read):
        mode = "accessr"
    else:
        mode = "accessw"
    if(remote_access):
        arrayLoc = 1
    else:
        arrayLoc = 0
    if(xCol == 2):
        for i in MEMS:
            x, yt, xm, ym  = parseFile(filepath, xCol, yCol, ["sweep", i, 1, mode, arrayLoc])
            ymax = max(ymax, ym)
            xmax = max(xmax, xm)
            realy.append(yt)
            realx.append(x)
    if(xCol == 1):
        for i in STRIDES:
            x, yt, xm, ym  = parseFile(filepath, xCol, yCol, ["sweep", 1, i, mode, arrayLoc])
            ymax = max(ymax, ym)
            xmax = max(xmax, xm)
            realy.append(yt)
            realx.append(x)

    print(realx)
    print(realy)
    my_plotter(ax[axx][axy], realx, realy, xmax, ymax, xlabel, ylabel, xCol==2)

fig, ax  = plt.subplots(4, 8, figsize=(80,40))

#Read, Default
callPlotter(ax, 0, 0, 2, 17, "H->D Transfer vs Stride with Read, Default Migration", "Stride Length", "Total H->D Memory Transfer (GB)", True, False)
callPlotter(ax, 0, 1, 2, 7, "Total Time  vs Stride with Read, Default Migration", "Stride Length", "Total Elapsed Time(s)", True, False)
callPlotter(ax, 1, 0, 2, 13, "H->D Number of Migrations vs Stride with Read, Default Migration", "Stride Length", "H->D Number of Migrations", True, False)
callPlotter(ax, 1, 1, 2, 33, "Total Size of GPU Page Fault Groups vs Stride with Read, Default Migration", "Stride Length", "Total Size of GPU Page Fault Groups (KB)", True, False)
callPlotter(ax, 2, 0, 1, 17, "H->D Transfer vs Problem Size with Read, Default Migration", "Problem Size", "Total H->D Memory Transfer (GB)", True, False)
callPlotter(ax, 2, 1, 1, 7, "Total Time vs Problem Size with Read, Default Migration", "Problem Size", "Total Elapsed Time(s)", True, False)
callPlotter(ax, 3, 0, 1, 13, "H->D Number of Migrations vs Problem Size with Read, Default Migration", "Problem Size", "H->D Number of Migrations", True, False)
callPlotter(ax, 3, 1, 1, 33, "Total Size of GPU Page Fault Groups vs Problem Size with Read, Default Migration", "Problem Size", "Total Size of GPU Page Fault Groups (KB)", True, False)

#Write, Default
callPlotter(ax, 0, 2, 2, 17, "H->D Transfer vs Stride with Write, Default Migration", "Stride Length", "Total H->D Memory Transfer (GB)", False, False)
callPlotter(ax, 0, 3, 2, 7, "Total Time  vs Stride with Write, Default Migration", "Stride Length", "Total Elapsed Time(s)", False, False)
callPlotter(ax, 1, 2, 2, 13, "H->D Number of Migrations vs Stride with Write, Default Migration", "Stride Length", "H->D Number of Migrations", False, False)
callPlotter(ax, 1, 3, 2, 33, "Total Size of GPU Page Fault Groups vs Stride with Write, Default Migration", "Stride Length", "Total Size of GPU Page Fault Groups (KB)", False, False)
callPlotter(ax, 2, 2, 1, 17, "H->D Transfer vs Problem Size with Write, Default Migration", "Problem Size (GB)", "Total H->D Memory Transfer (GB)", False, False)
callPlotter(ax, 2, 3, 1, 7, "Total Time vs Problem Size with Write, Default Migration", "Problem Size (GB)", "Total Elapsed Time(s)", False, False)
callPlotter(ax, 3, 2, 1, 13, "H->D Number of Migrations vs Problem Size with Write, Default Migration", "Problem Size (GB)", "H->D Number of Migrations", False, False)
callPlotter(ax, 3, 3, 1, 33, "Total Size of GPU Page Fault Groups vs Problem Size with Write, Default Migration", "Problem Size (GB)", "Total Size of GPU Page Fault Groups (KB)", False, False)

#Read, Remote Access
#callPlotter(ax, 0, 4, 2, 18, "H->D Transfer vs Stride with Read, Remote Access", "Stride Length", "Total H->D Memory Transfer (GB)", True, True)
callPlotter(ax, 0, 5, 2, 7, "Total Time  vs Stride with Read, Remote Access", "Stride Length", "Total Elapsed Time(s)", True, True)
callPlotter(ax, 1, 4, 2, 21, "Number of Remote Mappings vs Stride with Read, Remote Access", "Stride Length", "Number of Remote Mappings", True, True)
#callPlotter(ax, 1, 5, 2, 20, "Number of CPU Page Faults vs Stride with Read, Remote Access", "Stride Length", "Number of CPU Page Faults", True, True)
#callPlotter(ax, 2, 4, 1, 18, "H->D Transfer vs Problem Size with Read, Remote Access", "Problem Size (GB)", "Total H->D Memory Transfer (GB)", True, True)
callPlotter(ax, 2, 5, 1, 7, "Total Time vs Problem Size with Read, Remote Access", "Problem Size (GB)", "Total Elapsed Time(s)", True, True)
callPlotter(ax, 3, 4, 1, 21, "Number of Remote Mappings vs Problem Size with Read, Remote Access", "Problem Size (GB)", "Number of Remote Mappings", True, True)
#callPlotter(ax, 3, 5, 1, 20, "Number of CPU Page Faults vs Problem Size with Read, Remote Access", "Problem Size (GB)", "Number of CPU Page Faults", True, True)

#Write, Remote Access 
#callPlotter(ax, 0, 6, 2, 18, "H->D Transfer vs Stride with Write, Remote Access", "Stride Length", "Total H->D Memory Transfer (GB)", False, True)
callPlotter(ax, 0, 7, 2, 7, "Total Time  vs Stride with Write, Remote Access", "Stride Length", "Total Elapsed Time(s)", False, True)
callPlotter(ax, 1, 6, 2, 21, "Number of Remote Mappings  vs Stride with Write, Remote Access", "Stride Length", "Number of Remote Mappings", False, True)
#callPlotter(ax, 1, 7, 2, 20, "Number of CPU Page Faults vs Stride with Write, Remote Access", "Stride Length", "Number of CPU Page Faults", False, True)
#callPlotter(ax, 2, 6, 1, 18, "H->D Transfer vs Problem Size with Write, Remote Access", "Problem Size (GB)", "Total H->D Memory Transfer (GB)", False, True)
callPlotter(ax, 2, 7, 1, 7, "Total Time vs Problem Size with Write, Remote Access", "Problem Size (GB)", "Total Elapsed Time(s)", False, True)
callPlotter(ax, 3, 6, 1, 21, "Number of Remote Mappings vs Problem Size with Write, Remote Access", "Problem Size (GB)", "Number of Remote Mappings", False, True)
#callPlotter(ax, 3, 7, 1, 20, "Number of CPU Page Faults vs Problem Size with Write, Remote Access", "Problem Size (GB)", "Number of CPU Page Faults", False, True)







plt.tight_layout()
plt.savefig("figure.png")
plt.close(fig)
