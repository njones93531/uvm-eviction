import argparse
import os
import struct
import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.ticker import FixedLocator
matplotlib.use('Agg')
#matplotlib.use('pdf')
mplstyle.use('fast')


def load_data(file_path):
    print("loading data")
    with open(file_path, 'rb') as file:
        mode = ord(file.read(1))  # Read the mode from the first byte
        
        faults = []
        prefetch = []
        evictions = []
        address_ranges = []
       
        print(f"Plotting mode {mode} from magic byte")
        # mode 0 is simple; read the bytes and unpack them.
        if mode == 0:
            struct_format = '<QQHBBBBBBBBBBBBxx'
            record_size = struct.calcsize(struct_format)
            while True:
                binary_data = file.read(record_size)
                if not binary_data:
                    break
                data = struct.unpack(struct_format, binary_data)
                if data[-1] == 0:  # Fault
                    faults.append({'faddr': data[0]})
                elif data[-1] == 1:  # Prefetch
                    prefetch.append({'faddr': data[0]})
                elif data[-1] == 2:  # Eviction
                    evictions.append({'faddr': data[0]})
                elif data[-1] == 3:  # Address range
                    address_ranges.append({'base_address': data[0], 'length': data[1]})
                else:
                    print(f"Unknown record type detected, bailing out. ID was: {data[-1]}, {type(data[-1])}")
                    sys.exit(1)
        # mode 1 is harder; this is a more compressed format. 
        # Each record is 8 bytes, but address ranges are 16 bytes. That's what this code handles.
        # Because of this compression, the record type is encoded in the upper 8 bits of the address, because only the lower 50
        # bits are used in addressing. We parse out the record type and reset the address value to 0. In the case of faults,
        # this process is redundant, because their upper bytes are already 0 and their type is 0. Can improve speed marginally 
        # in this situation.
        elif mode == 1:
            struct_format = '<Q'
            record_size = struct.calcsize(struct_format)
            expecting_length = False
            base_address = None
            
            while True:
                binary_data = file.read(record_size)
                if not binary_data:
                    break
                fault_address, = struct.unpack(struct_format, binary_data)
                
                if not expecting_length:
                    record_type = fault_address >> 56
                    # Reset the upper byte only for fault address
                    fault_address &= 0x00FFFFFFFFFFFFFF  
                    #print(f"Fault Address: {hex(fault_address)}")
                    if record_type == 0:
                        faults.append({'faddr': fault_address})
                    elif record_type == 1:
                        prefetch.append({'faddr': fault_address})
                    elif record_type == 2:
                        evictions.append({'faddr': fault_address})
                    elif record_type == 3:
                        base_address = fault_address
                        expecting_length = True
                    else:
                        print(f"Unknown record type detected, bailing out. ID was: {data[-1]}, {typeof(data[-1])}")
                        sys.exit(1)
                else:
                    # The next data is length, expecting_length is True
                    length = fault_address
                    address_ranges.append({'base_address': base_address, 'length': length})
                    expecting_length = False

    print("converting data to Pandas")
    df_faults = pd.DataFrame(faults)
    df_prefetch = pd.DataFrame(prefetch)
    df_evictions = pd.DataFrame(evictions)
    df_address_ranges = pd.DataFrame(address_ranges)

# Filter address ranges to ensure they contain at least one fault
    print("adjusting address ranges")
    min_faddr = df_faults['faddr'].min()  # Calculate the minimum fault address
    valid_ranges = []
    for idx, range in df_address_ranges.iterrows():
        range_start = range['base_address']
        if range_start >= min_faddr:  # Check if range starts at or after min_faddr
            valid_ranges.append(range)
        else:
            print(f"Exclusiding allocation range starting at {hex(range_start)}")



    #TODO code below this checked every fault for inclusivity. That's too slow. Let's just filter out ranges below our address range,
    # which we believe are kernel entries
    #valid_ranges = []
    #for idx, range in df_address_ranges.iterrows():
    #    range_start = range['base_address']
    #    range_end = range['base_address'] + range['length']
    #    if any(range_start <= f['faddr'] <= range_end for index, f in df_faults.iterrows()):
    #        valid_ranges.append(range)

    df_valid_ranges = pd.DataFrame(valid_ranges)
    return df_faults, df_prefetch, df_evictions, df_valid_ranges

def plot_data(df_faults, df_prefetch, df_evictions, df_address_ranges, output_path, plot_end):

    print("configuring plot")
    global_min_address = df_address_ranges['base_address'].min() if not df_address_ranges.empty else 0
    df_faults['adjusted_faddr'] = df_faults['faddr'] - global_min_address
    df_address_ranges['adjusted_base'] = df_address_ranges['base_address'] - global_min_address
    df_address_ranges['adjusted_end'] = df_address_ranges['base_address'] + df_address_ranges['length']

    plt.figure(figsize=(10, 6))
    plt.scatter(df_faults.index, df_faults['adjusted_faddr'], color='green', alpha=0.5, label='Faults', marker='o', s=1, linewidths=0)

    if not df_prefetch.empty:
        df_prefetch['adjusted_faddr'] = df_prefetch['faddr'] - global_min_address
        plt.scatter(df_prefetch.index, df_prefetch['adjusted_faddr'], color='blue', alpha=0.5, label='Prefetches', marker='o', s=1, linewidths=0)
    if not df_evictions.empty:
        df_evictions['adjusted_faddr'] = df_evictions['faddr'] - global_min_address
        plt.scatter(df_evictions.index, df_evictions['adjusted_faddr'], color='purple', alpha=0.5, label='Evictions', marker='o', s=1, linewidths=0)
 


    #plt.scatter(df_faults.index, df_faults['adjusted_faddr'], color='blue', label='Faults', marker=',')
    xmin = 0
    xmax = max(df_faults.index.max(), len(df_faults) + len(df_address_ranges))

    for idx, row in df_address_ranges.iterrows():
        print(f"plotting allocation range starting at {hex(row['adjusted_base'])}")
        plt.hlines(y=row['adjusted_base'], xmin=xmin, xmax=xmax, color='black', label='Address Range Start' if idx == 0 else "")
        if plot_end:
            plt.hlines(y=row['adjusted_end'], xmin=xmin, xmax=xmax, color='red', label='Address Range End' if idx == 0 else "")
#    y_ticks = plt.gca().get_yticks()
#    plt.gca().set_yticklabels([hex(int(y)) for y in y_ticks])

    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels([hex(int(y)) for y in y_ticks])
    plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))


    plt.xlabel('Event Order')
    plt.ylabel('Address (Adjusted)')
    plt.legend()
    plt.tight_layout()
    print("plotting and saving")
    plt.savefig(output_path, format='png')
    #plt.savefig(output_path, format='pdf')

def main():
    parser = argparse.ArgumentParser(description="Process binary data into DataFrame and plot.")
    parser.add_argument("file_path", type=str, help="Path to the binary file")
    parser.add_argument("--plot_end", "-e", action='store_true', help="Plot the end of the address range as well")
    parser.add_argument("--output_path", "-o", type=str, help="Optional explicit output path")
    args = parser.parse_args()

    df_faults, df_prefetch, df_evictions, df_address_ranges = load_data(args.file_path)

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        output_path = args.output_path
        print(f"User specified output path: {output_path}")
    else:
        hostname = os.uname().nodename
        parts = args.file_path.split(os.sep)
        if ("dev_apps" in parts or "benchmarks" in parts) and any(part.startswith("log_") for part in parts):
            index = parts.index("dev_apps") if "dev_apps" in parts else parts.index("benchmarks")
            log_dir_index = next(i for i, part in enumerate(parts) if part.startswith("log_"))
            experiment_dir = parts[log_dir_index]  # This gets the directory like 'log_short_4096'
            benchmark = parts[index + 1]
            file_trail_digit = parts[-1].split('_')[-1]  # Assumes format 'something_digit'
            file_name = f"{experiment_dir}_{file_trail_digit}.png"
            output_dir = os.path.join("../../figs", hostname, "fault_plots", parts[index], benchmark)
            output_path = os.path.join(output_dir, file_name)
            print(f"Special condition path: {output_path}")
        else:
            output_dir = os.path.dirname(args.file_path)
            output_path = os.path.splitext(args.file_path)[0] + ".png"
            print(f"Default path (same directory as input file): {output_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_data(df_faults, df_prefetch, df_evictions, df_address_ranges, output_path, args.plot_end)
    print(f"Plot saved to {output_path}")



if __name__ == "__main__":
    main()

