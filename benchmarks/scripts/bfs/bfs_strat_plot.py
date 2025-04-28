#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import math

INFILE = "test2.in"
YLABEL = "Execution Time (s)"
XLABEL = "Problem Size"
TITLE = "Gaussian Access Policy vs Execution Time"

MARKERS = ["o", "v", "8", "s", "p", "P", "*", "h", "H", "X", "D", "^", "<", ">", "1", "2", "3", "4", "+", "x", "d"] #included for accessibility
PSIZES = [25,27.5,30,32.5,35,37.5,40] #PSIZES IN GB
STRATS = ["mmm","ppp","pmm","mpm","mmp","mpp","pmp","ppm"]


def parseFile(filepath):
    x = [[]]
    y = [[]]
    xmax = 0
    ymax = 0
    of = open(filepath, "r")
    lines = of.readlines()
    of.close()
    for i in STRATS:
        x.append([])
        y.append([])

    for line in lines:
        cols = line.split(",")
        for i in range(0, len(STRATS)):
            if STRATS[i]==cols[2].strip():
                x[i].append(float(cols[0]))
                y[i].append(float(cols[1]))
        xmax = max(xmax, float(cols[0]))
        ymax = max(ymax, float(cols[1]))
    return x, y, xmax, ymax

def div(x, N): #divide a 2d list by an integer (np cant help, inner lists have diff dimensions)
    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            x[i][j] = x[i][j] / N
    return x


def avgFiles(f1, f2, f3, f4, f5): #assumes input files are each in teh same order 
    N = 4
    xt, yt, xmax, ymax = parseFile(f1)
    x = div(xt,N)
    y = div(yt,N)
    for f in [f3, f4, f5]:
        tx, ty, txmax, tymax = parseFile(f)
        xmax = max(xmax, txmax)
        ymax = max(ymax, tymax)
        for i in range(0, len(tx)):
            for j in range(0, len(ty[i])):
                    y[i][j] = y[i][j] + (ty[i][j]/N)
                    x[i][j] = x[i][j] + (tx[i][j]/N)
    return x, y, xmax, ymax


def roundNice(num):
    if(num<=0):
        return num
    digits = math.log10(int(num))
    digits = math.floor(digits)
    factor = math.pow(10, digits)
    return round(float(num) / float(factor)) * factor

#make a generic plot
def my_plotter(ax, x, y, xmax, ymax, xlabel, ylabel):
    print("xmax, ymax " + str(xmax) + " " + str(ymax));
    print("xlen, ylen ", len(x), len(y))
    xmax = int(xmax * 1.2)
    ymax = int(ymax * 1.2)
    labels = []
    colors = []
    loopMax = 0 
    colors = cm.Dark2(np.linspace(0, 1, len(STRATS)))
    loopMax = len(STRATS)
    for i in range(0, loopMax):
        labels.append(STRATS[i])

    for i in range(0, loopMax):
        #ax.scatter(x[i], y[i], color=colors[i], marker=MARKERS[i], s=150, label=labels[i], alpha=0.7)
        ax.plot(x[i], y[i], color=colors[i], marker=MARKERS[i], label=labels[i], alpha=0.7)
    #xm_locator = roundNice(xmax/10)
    #ym_locator = roundNice(ymax/10)
    #ax.xaxis.set_major_locator(tkr.MultipleLocator(xm_locator))
    #ax.xaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    #ax.xaxis.set_minor_locator(tkr.MultipleLocator(xm_locator/4))
    #ax.yaxis.set_major_locator(tkr.MultipleLocator(ym_locator))
    #ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%d'))
    #ax.yaxis.set_minor_locator(tkr.MultipleLocator(ym_locator/4))
    ax.set_xlim(20000, xmax)
    ax.set_ylim(0, 3600)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.vlines(32768, 0, 10000, color='b', linestyle="dashed", label='Device Memory Capacity')
    ax.vlines(46340, 0, 10000, color='g', linestyle="dashed", label='100% Oversubscription')
    ax.vlines(56755, 0, 10000, color='r', linestyle="dashed", label='200% Oversubscription')
    ax.legend()
    ax.grid()

def callPlotter(ax, xCol, yCol, title, xlabel, ylabel):
    ax.set_title(title)
    filepath = INFILE #"formatted.out"
    x, y, xmax, ymax = parseFile(filepath)
    my_plotter(ax, x, y, xmax, ymax, xlabel, ylabel)

fig, ax  = plt.subplots(1, 1, figsize=(20,10))

callPlotter(ax, 0, 1, TITLE, XLABEL, YLABEL)
#x, y, xmax, ymax = avgFiles("1.txt","2.txt","3.txt","4.txt","5.txt")
#ax.set_title("Gaussian Access Strategy vs Execution Time")
#my_plotter(ax, x, y, xmax, ymax, "Problem Size", "Execution Time (s)")



plt.tight_layout()
plt.savefig("figure.png")
plt.close(fig)
