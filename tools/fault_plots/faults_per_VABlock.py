import argparse
import matplotlib.pyplot as plt 
import numpy as np
import time
import datetime
import sys

def main(args):
    filename = args.filename
    address_ranges = []

    #Key is alloc number, -1 is the whole allocation
    #-2 is an allocation not in ranges
    total_faults = {}
    total_migrations = {}
    min_addr = 10000000000000000000000000
    seen = {}
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')

            if len(parts) == 2 and parts[0].startswith('0x'):
                start_address = int(parts[0], 16)
                length = int(parts[1])
                address_ranges.append((start_address, start_address + length))
                if start_address < min_addr:
                    min_addr = start_address

            elif line.startswith('f,'):
                fault_address = int(parts[1], 16)
                
                aligned = (fault_address // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
                if aligned not in total_faults:
                    total_faults[aligned] = 0
                if aligned not in total_migrations:
                    total_migrations[aligned] = 0
                total_faults[aligned] += 1

                if aligned not in seen: 
                    seen[aligned] = None
                    total_migrations[aligned] += 1

            elif line.startswith('e,'):
                evicted_address = int(parts[1], 16)
                aligned = (evicted_address // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
                if aligned in seen:
                    del seen[aligned]
    
    x = []
    y = []
    ymax = 0
    for key in total_faults:
        faults = total_faults[key]
        migrations = total_migrations[key]
        fpm = faults / migrations
        x.append(key - min_addr)
        y.append(fpm)
        ymax = max(ymax, fpm)
    
    # Group data by 'App_Name' and plot y_column vs x_column for each group
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(x, y, color="blue", marker='.', alpha=0.8)
    
    for start, end in address_ranges:
        ax.vlines(start - min_addr, ymin=0, ymax=ymax, color="black")
        ax.vlines(end - min_addr, ymin=0, ymax=ymax, color="black")
    
    ax.set_xlabel("Relative VABlock Number")
    ax.set_ylabel("Faults per Migration")
    plt.savefig(args.outfile)
    plt.close('all')

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse a test file containing memory access traces, plot faults per migration for each VABlock')
    parser.add_argument('filename', metavar='F', type=str, help='The text file to be parsed.')
    parser.add_argument('outfile', type=str, help='What filename to save the output image')
    args = parser.parse_args()

    main(args)


