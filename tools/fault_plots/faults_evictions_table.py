import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import struct

from multiprocessing import Pool, cpu_count


SWAP_THROTTLE = cpu_count() - 1 
resolution = 300
format = 'png'
table_file = 'density_table.out'
max_input_bytes = 3865470566
axes_width_scalar = 2
axes_height_scalar = 4
max_cols = 5
ad = 0
tr = 1
dr = 2
ws = 3
sm_figs = []
label_size = 8
legend_size = 8
other_size = 8
all_scalar = 3
figsize=(4, 3)
location='best'

def prune_labels_allocs(labels, allocations):
    #First, group the allocs
    for i, l in reversed(list(enumerate(labels))):
        if '0GB' in l:
            i = i - 1
            if i > 0 and '0GB' in labels[i]:
                labels.pop(i)
    #Next, delete allocs with size 0
    for i, l in reversed(list(enumerate(labels))):
        if '0GB' in l:
            labels.pop(i)
            allocations = np.delete(allocations, i)
    return labels, allocations

def add_legend_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    return (mpatches.Patch(color=color), label)

def get_legend_order(labels):
    [int(x.split(' ')[-1], 16) for x in labels]
    sorted_labels.copy()
    sorted_labels.sort()
    return [labels.index(x) for x in sorted_labels]

def read_legend_label_lines():
    with open("legend_labels.conf") as file:
        return file.readlines()

def get_legend_labels(name):
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'.upper()
    name = name.upper()
    legend_label_lines = read_legend_label_lines()
    for i, line in enumerate(legend_label_lines):
        if name in line:
            labels = legend_label_lines[i+1].split(',')
            labels = [label.strip() for label in labels]
            #labels.reverse()
            return labels
    print(f"Name: {name} not found!")


def parse_file(file):
    with open(file, 'r') as f:
        lines = f.readlines(max_input_bytes)

    headers = {}
    faults = []
    evictions = []

    current_batch = []
    batch_id = 0

    for line in lines:
        #TODO remove me
        if ',' not in line:
            continue
        elif line[0] == 's':
            continue
        elif line[0] == 'b':
            continue
        elif line[0] == 'e':
            entries = line.strip().split(',')
            addr = int(entries[1], 16)
            entries[1] = addr
            for header in headers.keys():
                #greater than base address and less than range end
                if addr >= header and addr < header + headers[header]:
                    evictions.append([addr, header])
                    break
            else:
                print("Fault found with no matching allocation")

        elif line[0] == 'p':
            continue
        elif line[0] == 'f':
            entries = line.strip().split(',')
            addr = int(entries[1], 16)
            entries[1] = addr
            entries[2] = int(entries[2]) # timestamp
            for header in headers.keys():
                #greater than base address and less than range end
                if addr >= header and addr < header + headers[header]:
                    faults.append([addr, header])
                    break
            else:
                print("Fault found with no matching allocation")
        else:
            header = line.strip().split(',')
            # base address, allocation length
            # base addresses are base 16 but lengths are base 10 in bytes
            headers[int(header[0], 16)] = int(header[1], 10)
    return headers, faults, evictions

def get_output_file_name(input_file, specialization="", dirname=""):
    # Extract application_name and problem_size from the input file path
    application_name = os.path.basename(input_file).split("_")[0]
    problem_size = os.path.basename(os.path.dirname(input_file)).split("_")[5]

    # Define the output directory
    output_dir = os.path.join(f"../fault_plots/figures/metrics/{dirname}")
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file name
    output_file = os.path.join(output_dir, f"{application_name}-{problem_size}-density-{specialization}.{format}")

    return os.path.abspath(output_file)

def print_allocation_counts(faults_df, evicts_df, bmark):
    # Check if 'allocation' column is in the dataframe
    if 'allocation' not in faults_df.columns:
        print("Column 'allocation' not found in the DataFrame.")
        return

    unique_allocs = faults_df['allocation'].unique()
    unique_allocs.sort()
    unique_allocs = unique_allocs[::-1]
    # Iterate through the unique allocations and print the count of matching rows
    labels = get_legend_labels(bmark)
    for i, allocation in enumerate(unique_allocs):
        label = labels[i]
        faults = faults_df[faults_df['allocation'] == allocation]
        faults_count = len(faults)
        evicts = evicts_df[evicts_df['allocation'] == allocation]
        evicts_count = len(evicts)
        print(f"Bmark: {bmark}, alloc: {hex(allocation)}, label: {label}, faults: {faults_count}, evictions: {evicts_count}")




def main(args):
    output_dir = "../../figures/density/"
    os.makedirs(output_dir, exist_ok=True)
    num_processes = SWAP_THROTTLE
    if args.t: 
        num_processes = 1

    tasks = []
    with Pool(num_processes) as pool:
        pool.map(parse_and_construct_df, args.files)
        

def parse_and_construct_df(file):
    print(f"Parsing file: {file}")
    headers, faults, evictions = parse_file(file)  # Make sure parse_file is defined elsewhere in your code
    batch_cols = ["address", "allocation"]

    bmark = f"{file.split('/')[-2].split('_')[-1]}-{file.split('/')[-2].split('_')[-2]}"

    #faults_df = pd.concat([pd.DataFrame(fault columns=batch_cols, dtype=object) for fault in faults],
    #                       ignore_index=True)
    #evicts_df = pd.concat([pd.DataFrame(evict, columns=batch_cols, dtype=object) for evict in evictions],
    #                       ignore_index=True)

    faults_df = pd.DataFrame(faults, columns=batch_cols, dtype=object)
    evicts_df = pd.DataFrame(evictions, columns=batch_cols, dtype=object)
    print_allocation_counts(faults_df, evicts_df, bmark)  # Ensure that print_allocation_counts is defined elsewhere in your code


def execute_task(func, batches_df, headers, output_name, limit, *args):
    func(batches_df, headers, output_name, limit, args)

from datetime import datetime
import pytz

def get_time_str():
    # Define the timezone for EST
    est = pytz.timezone('US/Eastern')

    # Get the current UTC time and localize it to EST
    utc_now = datetime.now(pytz.utc)
    est_now = utc_now.astimezone(est)

    # Print the current time in EST
    return est_now.strftime('%Y-%m-%d %H:%M:%S %Z%z')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse text files containing memory access traces.')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='The text files to be parsed.')
    parser.add_argument('-t', action='store_true', help='Throttle the cpu count to avoid OOM')
    args = parser.parse_args()
    main(args)
