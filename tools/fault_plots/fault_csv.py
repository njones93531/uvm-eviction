import argparse
import numpy as np
import time
import datetime
import sys

batch_units = ['batches', 's', 'faults']
NS_PER_S = 1000000000

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_legend_label_lines():
    with open("legend_labels.conf") as file:
        return file.readlines()

def get_legend_labels(name):
    name = name.upper()
    legend_label_lines = read_legend_label_lines()
    for i, line in enumerate(legend_label_lines):
        if name in line:
            labels = legend_label_lines[i+1].split(',')
            labels = [label.strip() for label in labels]
            return labels
    print(f"Name: {name} not found!")

def dict_diff(a, b):
    c = {}
    for key in a:
        c[key] = a[key] - b[key]
    return c

#A is a list of dicts with the same keys
def dict_mean(a):
    from functools import reduce
    if len(a) == 0:
        a.append({"time":0, "local_faults":0, "faults":0, "batches":0})
    count = len(a)

    # sum the values with same keys
    total = reduce(lambda acc, d: {k: acc.get(k, 0) + d[k] for k in d}, a)
    # Calculate the average for each key
    return {k: v / count for k, v in total.items()} 





def mean(a):
    if len(a) == 0: 
        return 0
    return sum(a) / len(a)

def get_range(addr, ranges):
    for i, (start, end) in enumerate(sorted(ranges, key=lambda tup: tup[0], reverse=True)):
        if start <= addr < end:
            return i
    return -2

#Assumes that for a given application, once oversubscribed, allocations are always addressed
#in the same relative order
def set_alloc_names(filename):
    alloc_names = {-1:"ALL"}
    for i, label in enumerate(get_legend_labels(get_name(filename) + '-15')): 
        alloc_names[i] = label.split('-')[0].strip()
    return alloc_names

def set_start_addrs(ranges):
    start_addrs = {-1:0}
    for i, (start, end) in enumerate(sorted(ranges, key=lambda tup: tup[0], reverse=True)):
        start_addrs[i] = start
    return start_addrs

def set_alloc_sizes(alloc_size, total_faults):
    dec = 0
    for key, value in alloc_size.items():
        if total_faults[key] == 0:
            dec += 1
        else:
            alloc_size[key - dec] = value
    return alloc_size
            

def get_name(filename):
    return filename.split('/')[-2].split('_')[-1] 

def get_psize(filename):
    return filename.split('/')[-2].split('_')[-2] 

def parse_file(filename):
    address_ranges = []

    #Key is alloc number, -1 is the whole allocation
    #-2 is an allocation not in ranges
    total_faults = {-1:0}
    va_blocks_migrated = {-1:0}
    total_evictions = {-1:0}
    total_faultless_evictions = {-1:0}
    alloc_size = {-1:get_psize(filename)}
    offtime_between = {-1:[]}
    alltime_between = {-1:[]}

    
    total_faults[-2] = 0
    va_blocks_migrated[-2] = 0
    total_evictions[-2] = 0
    total_faultless_evictions[-2] = 0
    alloc_size[-2] = 1
    offtime_between[-2] = []
    alltime_between[-2] = []
    
    total_batches = 0
    seen = {}
    fseen = {}
    eseen = {}
    fault_time = {}
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')

            if len(parts) == 2 and parts[0].startswith('0x'):
                start_address = int(parts[0], 16)
                length = int(parts[1])
                address_ranges.append((start_address, start_address + length))
                i = len(address_ranges) -1
                total_faults[i] = 0
                va_blocks_migrated[i] = 0
                total_evictions[i] = 0
                offtime_between[i] = []
                alltime_between[i] = []
                total_faultless_evictions[i] = 0
                alloc_size[i] = round(length / (1024 * 1024 * 1024), 2)

            elif line.startswith('f,'):
                fault_address = int(parts[1], 16)
                i = get_range(fault_address, address_ranges)
                
                #Track the fault time 
                fault_time = {"time":int(parts[2], 10), "local_faults":total_faults[i], "faults":total_faults[-1], "batches":total_batches}
                
                total_faults[-1] += 1
                total_faults[i] += 1
                aligned = (fault_address // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
                if aligned not in seen: 
                    seen[aligned] = fault_time
                    va_blocks_migrated[-1] += 1
                    va_blocks_migrated[i] += 1
                if aligned in eseen:
                    
                    fault_time_delta = dict_diff(fault_time, eseen[aligned])
                    offtime_between[-1].append(fault_time_delta.copy())
                    offtime_between[i].append(fault_time_delta.copy())
                    del eseen[aligned]
                
                    if aligned in fseen: 
                        fault_time_delta = dict_diff(fault_time, fseen[aligned])
                        alltime_between[-1].append(fault_time_delta.copy())
                        alltime_between[i].append(fault_time_delta.copy())
                        fseen[aligned] = fault_time

                if aligned not in fseen:
                    fseen[aligned] = fault_time

            elif line.startswith('b,'):
                total_batches += 1
            elif line.startswith('e,'):
                evicted_address = int(parts[1], 16)
                i = get_range(evicted_address, address_ranges)
                total_evictions[-1] +=1
                total_evictions[i] +=1
                aligned = (evicted_address // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
                found = False
                if aligned in seen:
                    # Consider eviction as a reset for tracking
                    del seen[aligned]
                    #Use last known fault time
                    eseen[aligned] = fault_time.copy()
                    #Last known fault time has the wrong num local faults 
                    eseen[aligned]["local_faults"] = total_faults[i]
                else: 
                    #print(f"Address {hex(evicted_address)} evicted without fault?")
                    total_faultless_evictions[-1] +=1
                    total_faultless_evictions[i] +=1
    
    relevant = []
    for i in range(-1, len(address_ranges)):
        if total_faults[i] > 0:
            relevant.append(i);




    alloc_names = set_alloc_names(filename) 
    start_addrs = set_start_addrs(address_ranges)
    alloc_size = set_alloc_sizes(alloc_size, total_faults)

    for j, i in enumerate(relevant):
        offtime = dict_mean(offtime_between[i])
        alltime = dict_mean(alltime_between[i])
        print(get_name(filename),\
            get_psize(filename),\
            alloc_names[j-1],\
            alloc_size[i],\
            start_addrs[i],\
            total_faults[i],\
            va_blocks_migrated[i],\
            total_evictions[i],\
            total_faultless_evictions[i],\
            total_batches,\
            offtime["time"],\
            offtime["local_faults"],\
            offtime["faults"],\
            offtime["batches"],\
            alltime["time"],\
            alltime["local_faults"],\
            alltime["faults"],\
            alltime["batches"],\
            sep=',')       

def main(args):
    header = "App_Name,"+\
        "Problem_Size_(GB),"+\
        "Alloc_Name,"+\
        "Alloc_Size,"+\
        "Start_Address,"+\
        "Total_Faults,"+\
        "VABlocks_Migrated,"+\
        "Total_Evictions,"+\
        "Total_Faultless_Evictions,"+\
        "Total_Batches,"+\
        "Offtime_Between_(ns),"+\
        "Offtime_Between_(local_faults),"+\
        "Offtime_Between_(total_faults),"+\
        "Offtime_Between_(batches),"+\
        "Alltime_Between_(ns),"+\
        "Alltime_Between_(local_faults),"+\
        "Alltime_Between_(total_faults),"+\
        "Alltime_Between_(batches)"
    tstart = time.time()
    if args.print_header:
        print(header)
    for filename in args.files:
        parse_file(filename)
    tend = time.time()
    elapsed = str(datetime.timedelta(seconds=tend-tstart))
    eprint(f'Total processing time: {elapsed}') 



   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files containing memory access traces, and output csv of relevant statistics')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='The text files to be parsed.')
    parser.add_argument('-p', '--print_header', action='store_true', help='Print csv header')
    args = parser.parse_args()

    main(args)


