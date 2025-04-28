#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import math

INFILE = "3mm_x86_64-525.60.13_vanilla_10000_30000.data"
MARKERS = ["o", "v", "8", "s", "p", "P", "*", "h", "H", "X", "D", "^", "<", ">", "1", "2", "3", "4", "+", "x", "d"] #included for accessibility
STRATS = ["ppppppp","mmmmmmm","pppmmmm","mmmpppp","mmmmmmp"]
#STRATS = ["ABC Migrate","AB Migrate C Pin","A Placed B Migrate C Pin","A Placed BC Migrate","ABC Pin"]
#STRATS = ["ABC Migrate","A Migrate BC Pin", "A Pin BC Migrate", "ABC Pin", "AB Migrate C Pin", "AC Migrate B Pin", "A Placed B Migrate C Pin", "A Placed BC Migrate", "A Placed C Migrate B Pin", "A Placed BC Pin"]


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
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.vlines(21451, 0, ymax, color='b', linestyle="dashed", label='Device Memory Capacity')
    ax.vlines(30337, 0, ymax, color='g', linestyle="dashed", label='100% Oversubscription')
    ax.vlines(37155, 0, ymax, color='r', linestyle="dashed", label='200% Oversubscription')
    ax.legend()
    ax.grid()

def callPlotter(ax, xCol, yCol, title, xlabel, ylabel):
    ax.set_title(title)
    filepath = INFILE #"formatted.out"
    x, y, xmax, ymax = parseFile(filepath)
    my_plotter(ax, x, y, xmax, ymax, xlabel, ylabel)

fig, ax  = plt.subplots(1, 1, figsize=(20,10))

callPlotter(ax, 0, 1, "3MM Access Strategy vs Runtime", "Problem Size", "Runtime (s)")

plt.tight_layout()
plt.savefig("figure.png")
plt.close(fig)
