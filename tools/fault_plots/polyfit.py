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
POLY_DEGREE = 4

def get_name(output_file):
    name = f"{output_file.split('/')[-1].split('-')[0]}-{output_file.split('/')[-1].split('-')[1]}"
    if "TWITTER" in name.upper():
        name = 'spmv-coo-twitter7'
    return name

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

def get_legend_labels(name, suffix):
    name = name.upper()
    legend_label_lines = read_legend_label_lines()
    for i, line in enumerate(legend_label_lines):
        if name in line:
            labels = legend_label_lines[i+1].split(',')
            labels = [label.strip() + f' {suffix}' for label in labels]
            #labels.reverse()
            return labels
    print(f"Name: {name} not found!")


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
        return (total_instances - inter_duplicates + 1) # / unique_fault_addresses FIXME: Remember we decided dups / batch, not dups / fault / batch

    elif method == 'all':
        total_instances = group_data['num_instances'].sum()
        return (total_instances - unique_fault_addresses + 1) #/ unique_fault_addresses

    else:
        raise ValueError("Invalid method provided.")


def scatter_plot_polyfit(batches_df, headers, output_file, batchcap, linestyle, ax1, ax2, dataset_name, *args): #INUSE
    import numpy.polynomial.polynomial as poly

    allocations = batches_df['allocation'].unique()
    allocations.sort()
    allocations = allocations[::-1]
    name = get_name(output_file)
    labels = get_legend_labels(name, dataset_name)
    labels, allocations = prune_labels_allocs(labels, allocations)
    colors = plt.cm.Dark2.colors
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

    batchcap = batchcap // 2
    batches_df['first_batch_id'] = batches_df.groupby('allocation')['batch_id'].transform('min')
    batches_df['batch_group'] = (batches_df['batch_id'] - batches_df['first_batch_id']) // batchcap
    grouped_data = batches_df.groupby(['allocation', 'batch_group'])


    for idx, allocation in enumerate(allocations):
        print(f"Processing allocation: {hex(allocation)} with batchcap: {batchcap}...")
        batch_groups = []
        duplication_ratios = []
        unique_addresses = []
        access_density_values = []
        touchspan_ratio_values = []

        old_group_data = pd.DataFrame()
        old_group_datad = pd.DataFrame()
        for group_idx, ((alloc, batch_group), group_data_original) in enumerate(grouped_data):
            if alloc != allocation:
                continue

            group_data = pd.concat([group_data_original, old_group_data])
            group_datad = pd.concat([group_data_original, old_group_datad]).drop_duplicates(subset='fault_address', keep='first')

            smallest_page = group_datad['fault_address'].min() // 4096
            largest_page = group_datad['fault_address'].max() // 4096
            allocation_length = headers[allocation] // 4096

            page_range = largest_page - smallest_page + 1

            access_density = len(group_datad) / page_range
            access_density_values.append(access_density)

            touchspan_ratio = (page_range) / allocation_length
            touchspan_ratio_values.append(touchspan_ratio)

            batch_groups.append(batch_group)

            old_group_datad = group_data_original

            unique_address = compute_duplication_ratio(group_data, 'unique_addresses')
            duplication_ratio = compute_duplication_ratio(group_data, 'all')

            duplication_ratios.append(duplication_ratio)
            unique_addresses.append(unique_address)

            old_group_data = group_data_original

        unique_addresses = [u // batchcap for u in unique_addresses]
        duplication_ratios = [d / batchcap for d in duplication_ratios]
        current_color = colors[idx % len(colors)]
        current_marker = markers[idx % len(markers)]

        #scatter
        scatter_alpha = 0.01
        #ax1[0].scatter(batch_groups, duplication_ratios, marker=current_marker, color=current_color,
        #           alpha=scatter_alpha)
        #ax2[0].scatter(batch_groups, unique_addresses, marker=current_marker, color=current_color,
        #           alpha=scatter_alpha)
        #ax1[1].scatter(batch_groups, access_density_values, marker=current_marker, color=current_color,
        #            alpha=scatter_alpha)
        #ax2[1].scatter(batch_groups, touchspan_ratio_values, marker=current_marker, color=current_color,
        #            alpha=scatter_alpha)


        # Polyfit
        linealpha = 0.8
        linewidth = 4.0
        coefs = poly.polyfit(batch_groups, duplication_ratios, POLY_DEGREE)
        dup_ratio_fit = poly.polyval(batch_groups, coefs)    # instead of np.poly1d
        coefs = poly.polyfit(batch_groups, unique_addresses, POLY_DEGREE)
        unique_fit = poly.polyval(batch_groups, coefs)    # instead of np.poly1d
        coefs = poly.polyfit(batch_groups, access_density_values, POLY_DEGREE)
        access_density_fit = poly.polyval(batch_groups, coefs)    # instead of np.poly1d
        coefs = poly.polyfit(batch_groups, touchspan_ratio_values, POLY_DEGREE)
        touchspan_ratio_fit = poly.polyval(batch_groups, coefs)    # instead of np.poly1d
        percent_batches = batch_groups / max(batch_groups) * 100
        ax1[0].plot(percent_batches, dup_ratio_fit, color=current_color, linestyle=linestyle, linewidth=linewidth, label="label", alpha=linealpha)
        ax2[0].plot(percent_batches, unique_fit, color=current_color, linestyle=linestyle, linewidth=linewidth, label="label", alpha=linealpha)
        ax1[1].plot(percent_batches, access_density_fit, color=current_color, linestyle=linestyle, linewidth=linewidth, label="label", alpha=linealpha)
        ax2[1].plot(percent_batches, touchspan_ratio_fit, color=current_color, linestyle=linestyle, linewidth=linewidth, label="label", alpha=linealpha)

    return labels
       #fig.sup(f"{output_file.split('/')[-1].split('-')[0]}-{output_file.split('/')[-1].split('-')[1]}", fontsize=16)
    

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

def get_output_file_name(input_file, dir_base, specialization="", dirname=""):
    # Extract application_name and problem_size from the input file path
    application_name = os.path.basename(input_file).split("_")[0]
    problem_size = os.path.basename(os.path.dirname(input_file)).split("_")[-2]

    # Define the output directory
    output_dir = os.path.join(f"{dir_base}/{dirname}")
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file name
    output_file = os.path.join(output_dir, f"{application_name}-{problem_size}-density-{specialization}.{format}")

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
    batchcap = 20 
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(16, 12))
    name = get_output_file_name(args.files[0], args.outdir, f"density-{batchcap}-scatter-polyfit-all-metrics","polyfit")
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    dataset_name = ['pref', 'nopf']
    labels = []
    for i, file in enumerate(args.files):
        batches_df, headers, file = parse_and_construct_df(file)
        labels += scatter_plot_polyfit(batches_df, headers, name, batchcap, linestyles[i], ax1, ax2, dataset_name[i])
     

    for ax, density_name in [(ax1[0], "Fault Duplication Rate"), (ax2[0], "Working Set Delta"), (ax1[1], "Fault Density"), (ax2[1], "Fault Span Ratio")]:
        ax.tick_params(axis='both', labelsize=all_scalar*other_size)
        #ax.set_(f"Time vs. {density_name}")
        ax.set_xlabel('Time (% Batch Group)', fontsize=all_scalar*other_size)
        ax.set_ylabel(density_name, fontsize=all_scalar*other_size)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        hs, ls = ax.get_legend_handles_labels()
        ax.legend(hs, labels, framealpha=0.7, fontsize=all_scalar*legend_size, loc=location, labelspacing=0.3)

            
    plt.tight_layout()
    plt.savefig(f"{name[0:-4]}_all_metrics.{format}", format=format, dpi=resolution)
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

    return batches_df, headers, file


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
