import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def read_data(file_name):
    """Read CSV into a DataFrame."""
    return pd.read_csv(file_name)

def generate_plot(df, app_name):
    """Generate and save plot based on DataFrame."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns for the plots

    for case in ['vanilla', 'preempt', 'preempt-conservative']:
        subset = df[df['Case'] == case]

        subset = subset.sort_values(by='SizeLabel')

        mean_perfs = subset.groupby('SizeLabel')['Performance'].mean()
        std_perfs = subset.groupby('SizeLabel')['Performance'].std()
        axes[0].errorbar(mean_perfs.index, mean_perfs.values, yerr=std_perfs.values, fmt='-o', capsize=5, label=f'{case} Performance')

        axes[1].scatter(subset['SizeLabel'], subset['Performance'], marker='o', label=f'{case} Performance')

    axes[0].set_xlabel('Problem Size Label')
    axes[0].set_ylabel('Performance (GFLOP/s)')
    axes[0].set_title(f'{app_name} Performance with Error Bars')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('Problem Size Label')
    axes[1].set_ylabel('Performance (GFLOP/s)')
    axes[1].set_title(f'{app_name} Performance by Problem Size Label')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{app_name}_performance_combined_chart.png')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_chart.py <CSV file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    app_name = os.path.splitext(os.path.basename(csv_file))[0].split('_')[0]
    data = read_data(csv_file)
    generate_plot(data, app_name)

