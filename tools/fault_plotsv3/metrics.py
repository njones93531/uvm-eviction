import argparse
import json
import os

import fault_plot_functions
from fault_parsing import parse_fault_data, get_output_state_from_path_hostname, OutputState

fault_plot_functions_dict = {name: getattr(fault_plot_functions, name) for name in dir(fault_plot_functions)
                       if callable(getattr(fault_plot_functions, name)) and name.startswith('plot_')}
print(f"Plotting functions available: {fault_plot_functions_dict.keys()}")


def print_enabled_plots(config):
        print()
        print("############")
        print("The following plots are ENABLED by provided config:")
        for plot_name, is_enabled in config.items():
            if is_enabled:
                print(plot_name)
        print("############")
        print()

def main():
    parser = argparse.ArgumentParser(description="Process binary data into DataFrame and plot.")
    parser.add_argument("file_path", type=str, help="Path to the binary file")
    parser.add_argument("--plot_end", "-e", action='store_true', help="Plot the end of the address range as well")
    args = parser.parse_args()

    hostname, df_faults, df_prefetch, df_evictions, df_address_ranges = parse_fault_data(args.file_path)

    # Extract path parts for strategy and benchmark
    output_state = get_output_state_from_path_hostname(args.file_path, hostname)

    try:
        with open("default_plot_config.json", 'r') as file:
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
            fault_plot_functions_dict[plot_name](df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, args.plot_end)
            print(f"Finished {plot_name}")
            print("############")
            print()


if __name__ == "__main__":
    main()

