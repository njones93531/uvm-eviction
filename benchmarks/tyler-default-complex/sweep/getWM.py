#!/usr/bin/python3
import os
import numpy as np
import math

#reads all files from a given dir and prints the amount from the line 'VmHWM: <amount>'

directory = 'WMData'
files = []


def parseFile(filepath, filename):
    with open(filepath, "r") as mister_file:
        for line in mister_file:
            cols = line.split(':')
            if cols[0] == "VmHWM":
                print(filename.split('_')[2], ' ', cols[1])

for filename in os.listdir(directory):
    files.append(filename)
files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)))) #sort files based on the numbers in their names
for filename in files:
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        parseFile(f, filename)
        #print(f'Finished with file {f}\n')
