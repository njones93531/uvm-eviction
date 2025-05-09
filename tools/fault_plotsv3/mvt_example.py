import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import heapq
from itertools import combinations
from scipy.stats import chi2_contingency, gmean
import argparse

# Define the path to the main directory
perf_data_base_dir = '../../benchmarks/strategied'

def parse_perf_df():
    bmark = 'MVT'
    print("Parsing performance files")
    lc = bmark.lower()
    bench=f'{perf_data_base_dir}/{bmark}/{lc}_numa_pref.csv'
    data_df = pd.read_csv(bench)
    data_df['app'] = bmark
    data_df = data_df.drop(columns=['Iteration'])

    plt.plot(data_df['Problem Size'], data_df['Kernel Time'])
    # Plot
    textsize = 24
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(10, 6))
    
    plt.xlabel('Problem Size')
    plt.ylabel('Kernel Time')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(f"../../figs/voltron/oversub/", exist_ok=True)
    plt.savefig(f'../../figs/voltron/oversub/strategied_x86_64-555.42.02_faults-new_mvt.png')
    plt.close()

parse_perf_df()
