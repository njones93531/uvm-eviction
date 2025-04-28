import argparse
import json
import os
import struct
import sys
from collections import defaultdict

import pandas as pd

from fault_parsing import *
import fault_scaling_plot_functions

fault_scaling_plot_functions_dict = {name: getattr(fault_scaling_plot_functions, name) for name in dir(fault_scaling_plot_functions)
                       if callable(getattr(fault_scaling_plot_functions, name)) and name.startswith('plot_')}
print(f"Plotting functions available: {fault_scaling_plot_functions_dict.keys()}")

def print_enabled_plots(config):
        print()
        print("############")
        print("The following plots are ENABLED by provided config:")
        for plot_name, is_enabled in config.items():
            if is_enabled:
                print(plot_name)
        print("############")
        print()

# range_results is output
def compute_faults_per_vablock_migrated(experiment, range_results):

    all_range_ids = set()
    for i, row in experiment.df_address_ranges.iterrows():
        all_range_ids.add(i-1)

    #print(f"experiment.df_address_ranges {experiment.df_address_ranges}")
    df_faults = experiment.df_faults
    df_evictions = experiment.df_evictions
    df_address_ranges = experiment.df_address_ranges
    df_faults['bin'] = df_faults['adjusted_faddr'] // TWO_MB

    df_faults['range_id'] = None
    df_evictions['range_id'] = None
    for i, row in df_address_ranges.iterrows():
        mask = (df_faults['adjusted_faddr'] >= row['adjusted_base']) & (df_faults['adjusted_faddr'] < row['adjusted_end'])
        df_faults.loc[mask, 'range_id'] = i-1
        mask = (df_evictions['adjusted_faddr'] >= row['adjusted_base']) & (df_evictions['adjusted_faddr'] < row['adjusted_end'])
        df_evictions.loc[mask, 'range_id'] = i-1


    # Compute faults count
    faults_count = df_faults.groupby('range_id').size().reset_index(name='fault_count')

    # Compute base compulsory fault rate
    eviction_count = pd.DataFrame({
        'range_id': list(all_range_ids),
        'eviction_count': [df_faults[df_faults['range_id'] == i]['bin'].nunique() for i in all_range_ids]
    })

    # Count evictions in each range
    eviction_totals = df_evictions.groupby('range_id').size().reset_index(name='eviction_total_count')

    # Merge the two DataFrames on 'range_id'
    merged_df = pd.merge(eviction_count, eviction_totals, on='range_id', how='outer').fillna(0)

    # Sum the values by range_id and drop unused columns
    merged_df['total_eviction_count'] = merged_df['eviction_count'] + merged_df['eviction_total_count']
    result_df = merged_df[['range_id', 'total_eviction_count']]

    # Merge faults_count with result_df to match range_ids
    final_df = pd.merge(faults_count, result_df, on='range_id', how='outer').fillna(0)

    # Divide faults_count by total_eviction_count
    final_df['faults_per_migrated_region'] = final_df['fault_count'] / final_df['total_eviction_count']


    for _, row in final_df.iterrows():
        range_id = int(row['range_id'])
        print(f"range_id: {range_id}")
        faults_per_migrated_region = row['faults_per_migrated_region']
        range_results[range_id].append(faults_per_migrated_region)
    print("range_results", range_results)

    return range_results



def main():
    parser = argparse.ArgumentParser(description="Process binary data into DataFrame and plot.")
    parser.add_argument("file_paths", type=str, nargs='+', help="Path to the binary file")
    parser.add_argument("--plot_end", "-e", action='store_true', help="Plot the end of the address range as well")
    args = parser.parse_args()
    range_results = defaultdict(list)
    experiments = []

    for file_path in args.file_paths:

        # Extract path parts for strategy and benchmark
        path_parts = file_path.split('/')
        benchmark_index = path_parts.index('benchmarks') + 1
        strategy = path_parts[benchmark_index]
        benchmark = path_parts[benchmark_index + 1]
        dir_name = path_parts[benchmark_index + 2]

        hostname, df_faults, df_prefetch, df_evictions, df_address_ranges = parse_fault_data(file_path)
        output_state = OutputState(strategy, benchmark, dir_name, hostname)
        experiment = Experiment(file_path, df_faults, df_prefetch, df_evictions, df_address_ranges, output_state)
        compute_faults_per_vablock_migrated(experiment, range_results)
        
        experiments.append(experiment)
        del experiment.df_faults
        del experiment.df_prefetch
        del experiment.df_evictions

    print(range_results)
    
    try:
        with open("default_scaling_plot_config.json", 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise Exception(f"Fatal Error: The file {config_file} was not found.")
    except json.JSONDecodeError:
        raise Exception("Fatal Error: The file is not a valid JSON.")
   

    print_enabled_plots(config)

    # Execute plots based on the configuration
    for plot_name, should_plot in config.items():
        if should_plot:
            print()
            print("############")
            print(f"Starting {plot_name}")
            fault_scaling_plot_functions_dict[plot_name](experiments, args.plot_end, range_results)
            print(f"Finished {plot_name}")
            print("############")
            print()





if __name__ == "__main__":
    main()

