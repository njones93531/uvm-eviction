import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from functools import wraps
from matplotlib.ticker import FixedLocator

matplotlib.use('Agg')
#matplotlib.use('pdf')
mplstyle.use('fast')


def plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 6))
        result = func(*args, **kwargs)
        plt.clf()
        return result
    return wrapper

@plotter
def plot_faults_relative_order(df_faults, df_prefetch, df_evictions, df_address_ranges, output_state, plot_end):
    
    print("configuring plot")
    fontsize=20
    legend_marker_size=100
    linewidths=1.0
    output_path = output_state.get_output_path("faults_relative")
    plt.scatter([], [], color='green', alpha=1, label='Faults', marker='o', s=legend_marker_size, linewidths=linewidths)
    plt.scatter(df_faults.index, df_faults['adjusted_faddr'], color='green', alpha=0.1, label=None, marker='o', s=1, linewidths=linewidths)

    if not df_prefetch.empty:
        plt.scatter([],[], color='blue', alpha=1, label='Prefetches', marker='o', s=legend_marker_size, linewidths=linewidths)
        plt.scatter(df_prefetch.index, df_prefetch['adjusted_faddr'], color='blue', alpha=1, label=None, marker='o', s=1, linewidths=linewidths)
    if not df_evictions.empty:
        plt.scatter([],[], color='#ff7f0e', alpha=1, label='Evictions', marker='o', s=legend_marker_size, linewidths=linewidths)
        plt.scatter(df_evictions.index, df_evictions['adjusted_faddr'], color='#ff7f0e', alpha=1, label=None, marker='o', s=1, linewidths=linewidths)
 

    xmin = 0
    xmax = max(df_faults.index.max(), len(df_faults) + len(df_address_ranges))

    df_address_ranges = df_address_ranges.sort_values('adjusted_base', ascending=False).reset_index()
    skp = 0
    for idx, row in df_address_ranges.iterrows():
        print(f"plotting allocation relative range starting at {hex(int(row['adjusted_base']))}")
        plt.hlines(y=row['adjusted_base'], xmin=xmin, xmax=xmax, color='black', label='Address Range Start' if idx == 0 else "")
        if plot_end:
            plt.hlines(y=row['adjusted_end'], xmin=xmin, xmax=xmax, color='red', label='Address Range End' if idx == 0 else "")
        if row['length'] < 500000000:
            skp+=1
        else:
            midpoint = (row['adjusted_base'] + row['adjusted_end']) / 2
            plt.text(xmin, midpoint, chr(97 + idx - skp), ha='center', va='center', fontsize=24, color='red')
        
    from matplotlib.ticker import ScalarFormatter, FixedLocator
    ax = plt.gca()

    # Force scientific notation on both axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Optional: fix y-ticks if you want hex formatting
    y_ticks = ax.get_yticks()
    # ax.set_yticklabels([hex(int(y)) for y in y_ticks])  # Optional: hex labels
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))    # Fix the positions
    ax.get_yaxis().get_offset_text().set_visible(True)

    # Force tick label size
    plt.tick_params(axis='both', labelsize=fontsize)
    y_ticks = plt.gca().get_yticks()
    
    #plt.gca().set_yticklabels([hex(int(y)) for y in y_ticks])
    #plt.gca().yaxis.set_major_locator(FixedLocator(y_ticks))
    #plt.tick_params(axis='both', labelsize=fontsize)

    plt.xlabel('Event Order', fontsize=fontsize)
    plt.ylabel('Address (Adjusted)', fontsize=fontsize)
    #plt.legend()
    # Shrink current axis's height by 10% on the bottom
    box = plt.gca().get_position()
    plt.gca().set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), fancybox=False, shadow=False, ncol=5, fontsize=fontsize)


    print("plotting and saving")
    plt.savefig(output_path, format='png', bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("This is a utility class; you are probably looking for fault_plot.py")
