#!/usr/bin/python3
import os
import numpy as np
import math

#reads all files from a given dir and prints the relevant nvprof data into a csv

directory = 'toParse'
files = []

def formatFileToParse(filepath, filename): 
    csv = []
    with open(filepath, "r") as mister_file:
        csv = mister_file.readlines()
    with open(filepath, "w") as mister_file:
        for i in range(0, len(csv)):
            line = csv[i]
            mister_file.write(line)
            if line[:7] == " \"NVIDI":
                cols1=line[:-1].split(',')
                cols2=csv[i+1].split(',')
                if cols1[7] == "\"Host To Device\"" and cols2[7] != '"Device To Host"\n':
                    mister_file.write(" \"NVIDIA TITAN V (0)\",0,0,0,0,0,0,\"Device To Host\"\n")
                    

def parseFile(filepath, filename):
    with open(filepath, "r") as mister_file:
        args = []
        time = 0.0
        for line in mister_file:
            if line[:5] == "sweep":
                print()
                args = line.split('_')
                for i in [1,2,4]:
                    args[i] = ''.join(filter(str.isdigit, args[i])) #for strings with a number, get just the number
                argline = ""
                for i in range(0, len(args)):
                    argline += args[i]
                    argline += ","
                print(argline, end="")
            if line[:5] == " \"GPU":
                print(line[:-1], end="")
            #print('\n' + line[:5])
            if line[:7] == " \"NVIDI":
                cols=line[:-1].split(',')
                for i in range(0, len(cols)):
                    if 2 <= i and i < 7 and len(cols[i]) > 7:
                        cols[i] = cols[i][:-2]
                    print(cols[i], end=",")
                


for filename in os.listdir(directory):
    files.append(filename)
files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)))) #sort files based on the numbers in their names
for filename in files:
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        formatFileToParse(f, filename)
        parseFile(f, filename)
        #print(f'Finished with file {f}\n')
        print()
