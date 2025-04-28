import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from functools import wraps
from matplotlib.ticker import FixedLocator

import numpy as np

matplotlib.use('Agg')
#matplotlib.use('pdf')
mplstyle.use('fast')

# this is approximate which isn't great. need to make it part of the data-driven plot config somehow
GPU_MEM=12
TWO_MB = 2 * 1024 * 1024

def plotter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 6))
        result = func(*args, **kwargs)
        plt.clf()
        return result
    return wrapper

@plotter
def plot_faults_per_vablock_migrated(experiments, plot_end, range_results):

    print("configuring plot")
    output_path = experiments[0].output_state.get_output_path("faults_per_vablock_migrated")


    line_styles = ['-', '--', '-.', ':']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    marker_styles = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'H', '>', '<']

    total_ranges = len(range_results)
    needed_styles = total_ranges // len(line_styles) + 1
    line_styles *= needed_styles
    colors *= needed_styles
    marker_styles *= needed_styles

    x_labels = [int(100 * exp.output_state.psize/GPU_MEM) for exp in experiments]
    for i, (range_id, results) in enumerate(range_results.items()):
        style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        marker = marker_styles[i % len(marker_styles)]
        label = experiments[0].labels[range_id]  # Directly using labels from the experiment class

        print("x_labels: ", x_labels)
        print("results: ", results)
        plt.plot(x_labels, results, label=label, linestyle=style, color=color, marker=marker)

    plt.title('Faults per 2MB Region Migrated Across Experiments')
    plt.legend(title='Address Range')
    plt.xlabel('Percentage of Memory Subscription')
    plt.ylabel('Faults per Migrated VABlock')
    plt.tight_layout()
    print("plotting and saving")
    plt.savefig(output_path, format='png')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("This is a utility class; you are probably looking for fault_plot.py")
