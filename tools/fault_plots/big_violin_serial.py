import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import struct

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
title_size = 14
legend_size = 12
other_size = 14


def zip_wrapper(a, b):
    if len(a) > 1:
        return zip(a, b)
    if len(a) == 1:
        return [(a[0], b[0])]

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
            #labels.reverse()
            return labels
    print(f"Name: {name} not found!")


def add_legend_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    return (mpatches.Patch(color=color), label)

def get_legend_order(labels):
    labels = [int(x.split(' ')[-1], 16) for x in labels]
    sorted_labels = labels.copy()
    sorted_labels.sort()
    return [labels.index(x) for x in sorted_labels]

def compute_duplication_ratio(group_data, method):
    # Ensure 'num_instances' is treated as an integer
    group_data['num_instances'] = group_data['num_instances'].astype(int)

    unique_fault_addresses = group_data['fault_address'].nunique()

    if method == 'unique_addresses':
        return unique_fault_addresses

    elif method == 'inter':
        fault_counts = group_data['fault_address'].value_counts()
        duplicates = fault_counts.sum()
        return (duplicates) / unique_fault_addresses

    elif method == 'intra':
        # all faults
        total_instances = group_data['num_instances'].sum()
        # inter
        fault_counts = group_data['fault_address'].value_counts()
        inter_duplicates = fault_counts.sum()
        # all - inter
        return (total_instances - inter_duplicates + 1) / unique_fault_addresses

    elif method == 'all':
        total_instances = group_data['num_instances'].sum()
        return (total_instances - unique_fault_addresses + 1) #/ unique_fault_addresses #FIXME: We decided to do duplicates / batch, not dups / fault_addr

    else:
        raise ValueError("Invalid method provided.")

def violin_plot_big_touchspan_ratio(batches_df, headers, input_file, batchcap, metric, fig, ax): #INUSE
    allocations = batches_df['allocation'].unique()
    allocations.sort()
    allocations = allocations[::-1]
    name = f"{input_file.split('/')[-2].split('_')[-1]}-{input_file.split('/')[-2].split('_')[-2]}"
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'
    labels = get_legend_labels(name) #[hex(a) for a in allocations] 
    
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

    colors = plt.cm.Dark2.colors
    legend_labels = []
    
    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")
        batch_groups = []
        touchspan_ratio_values = []
        old_group_datad = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_datad = pd.concat([group_data_original, old_group_datad]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_datad['fault_address'].min() // 4096
            largest_page = group_datad['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            touchspan_ratio = (page_range) / allocation_length
            touchspan_ratio_values.append(touchspan_ratio)

            batch_groups.append(batch_group)

            old_group_datad = group_data_original

        width = 0.9
        showmedian = False
        showmean = True
        showextrema = True;
        legend_labels.append(add_legend_label(ax.violinplot([touchspan_ratio_values], [idx], widths=width, vert=True, showmeans=showmean, showextrema=showextrema, showmedians=showmedian), hex(allocation)))


    
    for ax, density_name in [(ax, "Fault Span Ratio")]:
        ax.set_title(name, fontsize=title_size)
        ax.tick_params(axis='y', labelsize=other_size)
        ax.set_xticks(np.arange(0, len(labels)), [])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        #ax.set_xlabel('Allocation')
        ls = ([x[0] for x in legend_labels], [x[1] for x in legend_labels])
        ax.legend(ls[0], labels, framealpha=0, fontsize=legend_size, loc='best')
    
    return fig, ax
    
    



def violin_plot_big_access_density(batches_df, headers, input_file, batchcap, metric, fig, ax): #INUSE
    allocations = batches_df['allocation'].unique()
    allocations.sort()
    allocations = allocations[::-1]
    name = f"{input_file.split('/')[-2].split('_')[-1]}-{input_file.split('/')[-2].split('_')[-2]}"
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'
    labels = get_legend_labels(name) #[hex(a) for a in allocations]
    
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


    print("Labels:")
    print(labels)
    print("Allocs:")
    print(allocations)

    colors = plt.cm.Dark2.colors
    legend_labels = []

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")
        batch_groups = []
        access_density_values = []
        old_group_datad = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_datad = pd.concat([group_data_original, old_group_datad]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_datad['fault_address'].min() // 4096
            largest_page = group_datad['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            access_density = len(group_datad) / page_range
            access_density_values.append(access_density)

            batch_groups.append(batch_group)

            old_group_datad = group_data_original

        width = 0.9
        showmedian = False
        showmean = True
        showextrema = True;
        legend_labels.append(add_legend_label(ax.violinplot([access_density_values], [idx], widths=width, vert=True, showmeans=showmean, showextrema=showextrema, showmedians=showmedian), hex(allocation)))



    for ax, density_name in [(ax, "Fault Density")]:
        ax.set_title(name, fontsize=title_size)
        ax.tick_params(axis='y', labelsize=other_size)
        ax.set_xticks(np.arange(0, len(labels)), [])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        #ax.set_xlabel('Allocation')
        ls = ([x[0] for x in legend_labels], [x[1] for x in legend_labels])
        ax.legend(ls[0], labels, framealpha=0, fontsize=legend_size, loc='best')
    
        return fig, ax
    
    
    
    

def violin_plot_big_duplication_rate(batches_df, headers, input_file, batchcap, metric, fig, ax): #INUSE
    allocations = batches_df['allocation'].unique()
    allocations.sort()
    allocations = allocations[::-1]
    name = f"{input_file.split('/')[-2].split('_')[-1]}-{input_file.split('/')[-2].split('_')[-2]}"
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'
    labels = get_legend_labels(name) #[hex(a) for a in allocations]
    
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


    colors = plt.cm.Dark2.colors
    legend_labels = []

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])


    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")
        batch_groups = []
        duplication_ratios = []
        old_group_data = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_data = pd.concat([group_data_original, old_group_data])

            batch_groups.append(batch_group)

            duplication_ratio = compute_duplication_ratio(group_data, 'all')

            duplication_ratios.append(duplication_ratio)

            old_group_data = group_data_original

        duplication_ratios = [d / batchcap for d in duplication_ratios]
        width = 0.9
        showmedian = False
        showmean = True
        showextrema = True;
        legend_labels.append(add_legend_label(ax.violinplot([duplication_ratios], [idx], widths=width, vert=True, showmeans=showmean, showextrema=showextrema, showmedians=showmedian), hex(allocation)))

    
    for ax, density_name in [(ax, "Fault Duplication Rate")]:
        ax.set_title(name, fontsize=title_size)
        ax.tick_params(axis='y', labelsize=other_size)
        ax.set_xticks(np.arange(0, len(labels)), [])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        #ax.set_xlabel('Allocation')
        ls = ([x[0] for x in legend_labels], [x[1] for x in legend_labels])
        ax.legend(ls[0], labels, framealpha=0, fontsize=legend_size, loc='best')
    return fig, ax
    
    
    
    

def violin_plot_big_working_set(batches_df, headers, input_file, batchcap, metric, fig, ax): #INUSE
    allocations = batches_df['allocation'].unique()
    allocations.sort()
    allocations = allocations[::-1]
    name = f"{input_file.split('/')[-2].split('_')[-1]}-{input_file.split('/')[-2].split('_')[-2]}"
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'
    labels = get_legend_labels(name) #[hex(a) for a in allocations]
    
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



    colors = plt.cm.Dark2.colors
    legend_labels = []

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])

    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")
        batch_groups = []
        unique_addresses = []
        old_group_data = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_data = pd.concat([group_data_original, old_group_data])
            batch_groups.append(batch_group)
            unique_address = compute_duplication_ratio(group_data, 'unique_addresses')
            unique_addresses.append(unique_address)
            old_group_data = group_data_original

        unique_addresses = [u // batchcap for u in unique_addresses]
        width = 0.9
        showmedian = False
        showmean = True
        showextrema = True;
        legend_labels.append(add_legend_label(ax.violinplot([unique_addresses], [idx], widths=width, vert=True, showmeans=showmean, showextrema=showextrema, showmedians=showmedian), hex(allocation)))

    for ax, density_name in [(ax, "Working Set Delta")]:
        ax.set_title(name, fontsize=title_size)
        ax.tick_params(axis='y', labelsize=other_size)
        ax.set_xticks(np.arange(0, len(labels)), [])
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ls = ([x[0] for x in legend_labels], [x[1] for x in legend_labels])
        ax.legend(ls[0], labels, framealpha=0, fontsize=legend_size, loc='best')
    return fig, ax


def violin_plot_big(batches_df, headers, input_file, batchcap, metric, i, j, k, rows, cols): #INUSE
     

    #Read fig and axis from shared array
    fig = sm_figs[k][j]                    
    ax = fig.axes
    if rows > 1 and isinstance(ax, list):
        ax = ax[i%cols + (cols* (i//cols))]
    if isinstance(ax, list):
        ax = ax[i % cols]

    if metric == ad:
        fig, ax = violin_plot_big_access_density(batches_df, headers, input_file, batchcap, metric, fig, ax)
    elif metric == tr:
        fig, ax = violin_plot_big_touchspan_ratio(batches_df, headers, input_file, batchcap, metric, fig, ax)
    elif metric == dr:
        fig, ax = violin_plot_big_duplication_rate(batches_df, headers, input_file, batchcap, metric, fig, ax)
    elif metric == ws:
        fig, ax = violin_plot_big_working_set(batches_df, headers, input_file, batchcap, metric, fig, ax)
    else:
        print("Bad metric passed to violin_plot_big")
    #Write fig and ax back in shared array 
    sm_figs[k][j] = fig

def parse_file(file):
    with open(file, 'r') as f:
        lines = f.readlines(max_input_bytes)

    headers = {}
    batches = []

    current_batch = []
    batch_id = 0

    for line in lines:
        #TODO remove me
        if ',' not in line:
            continue
        elif line[0] == 's':
            current_batch = []
        elif line[0] == 'b':
            batches.append(current_batch)
            batch_id += 1
        elif line[0] == 'e' or line[0] == 'p':
            continue
        elif line[0] == 'f':
            entries = line.strip().split(',')
            entries.append(batch_id)
            addr = int(entries[1], 16)
            entries[1] = addr
            entries[2] = int(entries[2]) # timestamp
            for header in headers.keys():
                #greater than base address and less than range end
                if addr >= header and addr < header + headers[header]:
                    entries.append(header)
                    break
            else:
                print("Fault found with no matching allocation")
            current_batch.append(entries[1:])  # Exclude the leading 'f'
        else:
            header = line.strip().split(',')
            print(header)
            # base address, allocation length
            # base addresses are base 16 but lengths are base 10 in bytes
            headers[int(header[0], 16)] = int(header[1], 10)
    print(f"headers: {headers}")
    return headers, batches

def get_output_file_name(output_file, dir_base, specialization="", dirname=""):
    # Extract application_name and problem_size from the input file path
    base = output_file.split('/')[-1].split('.')[0]

    # Define the output directory
    output_dir = os.path.join(f"{dir_base}/{dirname}")
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file name
    output_file = os.path.join(output_dir, f"{base}-density-{specialization}.{format}")

    return os.path.abspath(output_file)




def print_allocation_counts(df):
    # Check if 'allocation' column is in the dataframe
    if 'allocation' not in df.columns:
        print("Column 'allocation' not found in the DataFrame.")
        return

    unique_allocs = df['allocation'].unique()
    print(f"{len(unique_allocs)}")
    # Iterate through the unique allocations and print the count of matching rows
    for allocation in unique_allocs:
        count = len(df[df['allocation'] == allocation])
        print(f"Allocation: {allocation}, Number of matching rows: {count}")

def main(args):
    output_dir = "../../figures/density/"
    os.makedirs(output_dir, exist_ok=True)

    input_files = []
    if len(args.files) == 1:
        with open(args.files[0], "r") as f:
            input_files = f.readlines()
        input_files = [line.rstrip('\n') for line in input_files]
    else: 
        return
    rows = max(len(input_files) // max_cols, 1)
    cols = min(max_cols, len(input_files))
    batchcaps = [20]#, 50, 100, 200]
    metric_names = {ad:"Fault Density", tr:"Fault Span Ratio", dr:"Fault Duplication Rate", ws:"Working Set Delta"}
    
    for i, metric in enumerate(metric_names):
        sm_figs.append([])
        for j, batchcap in enumerate(batchcaps):
            f, a = plt.subplots(rows, cols, sharey=True, figsize=(cols * axes_width_scalar,rows * axes_height_scalar))
            sm_figs[i].append(f)

    print(sm_figs)

    for i, fpath in enumerate(input_files):
        tasks = []
        results = [parse_and_construct_df(fpath)]
        for batches_df, headers, file in results:
            for j, batchcap in enumerate(batchcaps):
                for k, metric in enumerate(metric_names):
                    violin_plot_big(batches_df, headers, fpath, batchcap, metric, i, j, k, rows, cols)
        
    for j, batchcap in enumerate(batchcaps):
        for k, metric in enumerate(metric_names):
            
            if isinstance(sm_figs[k][j].axes, list):
                sm_figs[k][j].axes[0].set_ylabel(metric_names[metric], fontsize=other_size)
                #for a in sm_figs[k][j].axes[2:]:
                #    a.set_yticks([])
            else:
                sm_figs[k][j].axes.set_ylabel(metric_names[metric], fontsize=other_size)
            sm_figs[k][j].tight_layout()
            sm_figs[k][j].subplots_adjust(wspace=0)
            sm_figs[k][j].savefig(get_output_file_name(args.files[0], args.outdir, f"metrics-{batchcap}-violin-big-{metric_names[metric].replace(' ', '_')}",
                                "violin/violin-big"), format=format, dpi=resolution)
    plt.close()

def parse_and_construct_df(file):
    print(f"Parsing file: {file}")
    headers, batches = parse_file(file)  # Make sure parse_file is defined elsewhere in your code
    batch_cols = [
        "fault_address", "timestamp", "fault_type", "fault_access_type",
        "access_type_mask", "num_instances", "client_type", "mmu_engine_type",
        "client_id", "mmu_engine_id", "utlb_id", "gpc_id", "channel_id",
        "ve_id", "batch_id", "allocation"
    ]

    batches_df = pd.concat([pd.DataFrame(batch, columns=batch_cols, dtype=object) for batch in batches],
                           ignore_index=True)
    print_allocation_counts(batches_df)  # Ensure that print_allocation_counts is defined elsewhere in your code

    result = (batches_df, headers)
    return (batches_df, headers, file)


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
    parser.add_argument('-o', '--outdir', help='Output directory base', default="../fault_plots/figures/metrics")
    args = parser.parse_args()
    print(f"starting at {get_time_str()}")
    main(args)
    print(f"finishing at {get_time_str()}")
