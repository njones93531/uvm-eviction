import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
import heapq
import config
from itertools import combinations
from scipy.stats import chi2_contingency, gmean
import argparse
import socket

# Define the path to the main directory
hostname=socket.gethostname()
main_directory = f'../../figs/{hostname}/metrics_stats_relative/'
print(f"Main Directory: {main_directory}")
perf_data_base_dir = '../../benchmarks/strategied'
metric_stats_type = 'default'
kernel_version = 'x86_64-555.42.02'
psizes = [100, 125, 150, 175]
benchmarks = ['FDTD-2D', 'GRAMSCHM', 'stream', 'sgemm', 'bfs-worst', 'tealeaf', 'conjugateGradientUM']
PAPER_METS = ['app', 'psize', 'label', 'size', 'tr_median_1000', 'ts_rel_median_1000', 'tr_median_1000_OR_ts_rel_median_1000', 'd_mean_1000', 'dc_intra_rel_mean_1', 'dr_intra_mean_1000', 'dr_intra_mean_1', 'tr_mean_1', 'ws_mean_1_OVER_size']
#Fontsize for radar chart
base=48 
pd.options.display.max_rows = 999
pd.set_option('display.max_columns', 999)
enable_good_OR = True
enable_OSOR = False
enable_OR = False
enable_RATIO = False
ENABLE_DR_COMBOS = False

solution = pd.DataFrame({
        'app': ['bfs-worst', 'sgemm', 'FDTD-2D', 'GRAMSCHM', 'stream'],
        #9.0  : ['ddd', 'ddd', 'ddd', 'ddd', 'ddd'],
        100 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd'],
        125 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd'],
        150 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm'],
        175 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm']}).set_index('app')
    

def parse_perf_df():
    print("Parsing performance files")
    dfs = []
    for bmark in benchmarks:
        lc = bmark.lower()
        bench=f'{perf_data_base_dir}/{bmark}/{lc}_numa_pref.csv'
        data_df = pd.read_csv(bench)
        data_df['app'] = bmark
        data_df = data_df.drop(columns=['Iteration'])
        dfs.append(data_df)
    # Concatenate all the DataFrames together
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['Problem Size'] = round(100 * combined_df['Problem Size'] / config.VRAM_SIZE)
    print("Finished parsing performance files")
    return combined_df

def df_corr(df):
    ## Convert pandas DataFrame to Dask DataFrame
    #ddf = dd.from_pandas(df.copy(), npartitions=8)

    ## Compute correlation matrix in parallel
    #corr_matrix = ddf.corr().compute()
    #clean_df = df.loc[:, (df != df.iloc[0]).any()]
    corr_matrix = df.corr()
    return corr_matrix

def df_chi(df):
    # Initialize an empty DataFrame to store chi-squared values
    chi_squared_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Compute chi-squared values for each pair of columns
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 == col2:
                chi_squared_matrix.loc[col1, col2] = np.nan  # NaN for diagonal
            else:
                # Create a contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])
                # Perform the chi-squared test
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi_squared_matrix.loc[col1, col2] = p

    # Convert to numeric
    chi_squared_matrix = chi_squared_matrix.astype(float)
    
    return chi_squared_matrix

def corr(df, col1, col2):
    if df[col1].nunique() != 1 and df[col2].nunique() != 1:
        return df[col1].corr(df[col2])
    else:
        return 0

def chi(df, col1, col2):
    if col1 == col2:
        return np.nan
    else:
        # Create a contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        # Perform the chi-squared test
        chi2, p, _, _ = chi2_contingency(contingency_table)
        return p

    

def migrate_me(df, cap, cost_str, val_str):
    # Start with all costs and values
    costs = df[cost_str].values
    values = df[val_str].values
    
    # Create a temporary DataFrame and sort by value
    tmp = df.copy()
    tmp = tmp.sort_values(val_str, ascending=False)  # Sort descending for highest value first
    
    # Initialize a list to keep track of indices of items on the device
    on_device = []

    # Remove allocations from the device until undersubscribed
    while sum(costs) > cap and not tmp.empty:
        # Get the index of the item with the lowest value
        i = tmp.index[-1]  # Get the last index (smallest value in sorted DataFrame)
        costs[i] = 0  # Set the cost to 0 to "remove" it
        tmp = tmp[:-1]  # Remove this item from the temporary DataFrame

    # Collect indices of items still on the device
    for i in range(len(costs)):
        if costs[i] > 0:
            on_device.append(i)

    return on_device

#def migrate_me(df, cap, cost_str, val_str):
#    #Start with all on 'd'
#    costs = df[cost_str].values
#    values = df[val_str].values
#    tmp = df.copy()
#    tmp = tmp.sort_values(val_str)
#    tmp['index'] = tmp['index'].astype(int)
#    print(tmp)
#    #Remove allocs from device until undersubscribed
#    i = 0
#    while(sum(costs) > cap):
#        costs[tmp.iloc[i,'index']] = 0
#
#    on_device = []
#    for i, e in costs:
#        if e > 0:
#            on_device.append(i)
#    return on_device

        
def knapsack(df, cap, cost_str, val_str, threshold=0.02):
    """
    Solve the knapsack problem using a branch-and-bound approach, 
    with a heuristic to pre-include low-cost items.

    Args:
        capacity (float): Maximum cost allowed.
        costs (list of float): List of item costs.
        values (list of float): List of item values.
        threshold (float): Cost threshold for heuristic inclusion.

    Returns:
        list of int: Indexes of items to include in the knapsack.
    """
    capacity = cap
    costs = df[cost_str].values
    values = df[val_str].values
    n = len(costs)


    # Heuristically include items with cost < threshold
    low_cost_items = [i for i in range(n) if costs[i] < threshold]
    included_items = low_cost_items[:]
    remaining_capacity = capacity - sum(costs[i] for i in low_cost_items)
    total_value = sum(values[i] for i in low_cost_items)

    # Filter remaining items
    remaining_items = [(i, costs[i], values[i]) for i in range(n) if i not in low_cost_items]
    if not remaining_items or remaining_capacity <= 0:
        return included_items

    # Sort remaining items by value-to-cost ratio
    items = sorted(remaining_items, key=lambda x: -x[2] / x[1])

    # Max heap for branch-and-bound
    max_heap = []
    best_value = total_value
    best_solution = included_items

    # Push the root node
    heapq.heappush(max_heap, (-total_value, 0, 0, total_value, included_items))

    while max_heap:
        neg_value, level, current_cost, current_value, solution = heapq.heappop(max_heap)
        neg_value = -neg_value

        if level == len(items) or current_cost > remaining_capacity:
            continue

        # Update the best solution
        if current_value > best_value:
            best_value = current_value
            best_solution = solution

        # Get the current item
        index, cost, value = items[level]

        # Include the current item (if possible)
        if current_cost + cost <= remaining_capacity:
            heapq.heappush(
                max_heap,
                (-neg_value - value, level + 1, current_cost + cost, current_value + value, solution + [index])
            )

        # Exclude the current item
        heapq.heappush(
            max_heap,
            (-neg_value, level + 1, current_cost, current_value, solution)
        )

    return best_solution

def knapsack3(df, cap, cost_str, val_str):
    capacity = cap
    costs = df[cost_str].values
    values = df[val_str].values
    n = len(costs)

    # Calculate value-to-cost ratios
    items = sorted([(i, values[i] / costs[i], costs[i], values[i]) for i in range(n)], key=lambda x: -x[1])

    # Max heap for branch-and-bound
    max_heap = []
    best_value = 0
    best_solution = []

    # Push the root node
    heapq.heappush(max_heap, (-0, 0, 0, 0, []))  # (-neg_value, level, current_cost, current_value, included_items)

    while max_heap:
        neg_value, level, current_cost, current_value, included_items = heapq.heappop(max_heap)
        neg_value = -neg_value

        if level == n or current_cost > capacity:
            continue

        # Update the best solution if this node has a better value
        if current_value > best_value:
            best_value = current_value
            best_solution = included_items

        # Get the current item
        index, ratio, cost, value = items[level]

        # Include the current item (if possible)
        if current_cost + cost < capacity:
            heapq.heappush(
                max_heap,
                (-neg_value - value, level + 1, current_cost + cost, current_value + value, included_items + [index])
            )

        # Exclude the current item
        heapq.heappush(
            max_heap,
            (-neg_value, level + 1, current_cost, current_value, included_items)
        )

    return best_solution

def knapsack2(df, cap, cost_str, val_str):
    # Extract costs and values from the DataFrame
    costs = df[cost_str].values
    values = df[val_str].values
    #print("costs", costs)
    #print("values", values)
    
    # Number of items
    n = len(df)
    
    # Track the best value and the corresponding set of indexes
    best_value = 0.0
    best_combination = []

    # If all costs are equal, answer is trivial
    if df[cost_str].nunique() == 1: 
        cost = costs[0]
        num_to_choose = int(np.floor(cap / cost))
        return np.argpartition(values, -num_to_choose)[-num_to_choose:]
    
    # Loop over all possible combinations of items
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            total_cost = sum(costs[i] for i in combo)
            total_value = sum(values[i] for i in combo)
            # Check if the current combination is within the capacity
            if total_cost <= cap and total_value > best_value:
                best_value = total_value
                best_combination = combo
    
    # Return the indexes of the items in the best combination
    return list(best_combination)

def check_strat(s1, s2):
    if len(s1) != len(s2):
        return False
    for i, c1 in enumerate(s1):
        c2 = s2[i]
        if (c1 == 'd' or c2 == 'd') and c1 != c2:
            return False
        if not (c1 == '-' or c2 == '-') and c1 != c2:
            return False
    return True

def print_x_classification(data_df, metric):
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group.sort_values('label').reset_index(drop=True)
        classification = '-'
        # Heuristic-ify the strategy
        #If threshold is high, use t
        #d first
        if tmp['predict_x'].all():
        #if True:
            classification = 'x'
            #classification = f"{tmp[metric].iloc[0]}"

        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'x_class'] = str(classification)

    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])                                                              # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='x_class').sort_values(by='app', key=lambda x: x.str.lower())
    #print(f"Using metric {metric}")
    #print(pivot_table)

def get_solution(apps, psizes, perf_df):
    # Create an empty DataFrame to store the comparison results
    optimal_strategies = pd.DataFrame("", index=apps, columns=psizes)
    optimal_strategies.index.name = 'app'
    
    #Problem Size,Policy,Iteration,Kernel Time
    # Iterate through each app and psize and compare the strategy strings
    for app in apps:
        for psize in psizes:
            cell_view = None
            no_value = False
            try:
                cell_view = perf_df[(perf_df['app'] == app) & (perf_df['Problem Size'] == int(psize))].groupby(['app', 'Problem Size', 'Policy'], observed=True).mean().reset_index()
                cell_view_best_time = float(cell_view[cell_view['Kernel Time'] > 0.01]['Kernel Time'].min())
                cell_view_best_policy = str(cell_view[cell_view['Kernel Time'] == cell_view_best_time]['Policy'].iloc[0])
            except (IndexError, KeyError):
                no_value = True
                
            if not no_value: 
                optimal_strategies.loc[app, psize] = cell_view_best_policy
            
    return optimal_strategies



def compare_pivot_tables_speedup(pivot1, perf_df):
    """
    Compare two pivot tables and return a new pivot table with
    the lost speedup compared to the optimal policy, relative to the default

    Parameters:
    - pivot1: First pivot table (DataFrame)
    - pivot2: Second pivot table (DataFrame)

    Returns:
    - A new pivot table with float for each (app, psize) combination.
    """
    # Get all the apps (row labels) and psize (column labels) from the pivot tables
    apps = pivot1.index
    psizes1 = pivot1.columns

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(0.0, index=apps, columns=psizes1)
    comparison_result.index.name = 'app'
    comparison_default = pd.DataFrame(0.0, index=apps, columns=psizes1)
    comparison_default.index.name = 'app'
    score = [0, 0]
    
    #Problem Size,Policy,Iteration,Kernel Time
    # Iterate through each app and psize and compare the strategy strings
    for app in apps:
        for psize in psizes1:
            strategy1 = ""
            cell_view = None
            no_value = False
            try:
                strategy1 = str(pivot1.loc[app, psize])
                cell_view = perf_df[(perf_df['app'] == app) & (perf_df['Problem Size'] == int(psize))].groupby(['app', 'Problem Size', 'Policy'], observed=True).mean().reset_index()
                cell_view_default_time = float(cell_view[cell_view['Policy'] == 'm' * len(strategy1)]['Kernel Time'].iloc[0])
                cell_view_best_time = float(cell_view[cell_view['Kernel Time'] > 0]['Kernel Time'].min())
                cell_view_best_policy = str(cell_view[cell_view['Kernel Time'] == cell_view_best_time]['Policy'].iloc[0])
                cell_view_predicted_time = 0
                if '-' in strategy1:
                    policies_view = cell_view[cell_view['Policy'].apply(lambda policy1: check_strat(strategy1, policy1))]
                    cell_view_predicted_time = float(policies_view[policies_view['Kernel Time'] > 0]['Kernel Time'].min())
                else:
                    cell_view_predicted_time = float(cell_view[cell_view['Policy'] == strategy1]['Kernel Time'].iloc[0])
            except IndexError:
                no_value = True
            except KeyError:
                no_value = True
                
            # Store True if strategies match, else False
            if no_value: 
                comparison_result.loc[app, psize] = 2.0
                comparison_default.loc[app, psize] = 2.0
            else:
                result = (cell_view_predicted_time - cell_view_best_time) / cell_view_best_time
                default = cell_view_default_time / cell_view_predicted_time
                comparison_result.loc[app, psize] = result
                comparison_default.loc[app, psize] = default
                score[0] += result
                score[1] = max(score[1], result)
            
    comparison_result = comparison_result.sort_values(by='app', key=lambda x: x.str.lower())
    comparison_default = comparison_default.sort_values(by='app', key=lambda x: x.str.lower())
    return comparison_result, comparison_default, score



def compare_pivot_tables(pivot1, pivot2):
    """
    Compare two pivot tables and return a new pivot table with True/False
    indicating whether the strategies match for each (app, psize) combination.

    Parameters:
    - pivot1: First pivot table (DataFrame)
    - pivot2: Second pivot table (DataFrame)

    Returns:
    - A new pivot table with True/False for each (app, psize) combination.
    """
    # Get all the apps (row labels) and psize (column labels) from the pivot tables
    apps = pivot1.index
    psizes = pivot1.columns

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(False, index=apps, columns=psizes)
    comparison_result.index.name = 'app'
    score = 0
    pp = psizes#[12.0, 15.0]
    aa = apps#['stream', 'sgemm']#'bfs-worst', 'FDTD-2D', 'GRAMSCHM']#'sgemm']
    goals = [(12.0, 'FDTD-2D'), (12.0, 'stream'), (15.0, 'FDTD-2D'), (15.0, 'stream')]
    # Iterate through each app and psize and compare the strategy strings
    for app in apps:
        for psize in psizes:
            strategy1 = str(pivot1.loc[app, psize])
            strategy2 = str(pivot2.loc[app, psize])
            
            # Store True if strategies match, else False
            if check_strat(strategy1, strategy2):
                comparison_result.loc[app, psize] = True
                score+=1

           # if (psize, app) in goals:
           #     for i in range(0, len(strategy1)):
           #         if strategy1[i] == strategy2[i]:
           #             score+=1
           #         else:
           #             score+=0.01
    comparison_result = comparison_result.sort_values(by='app', key=lambda x: x.str.lower())

    return comparison_result, score

def parse_df():
    print("Parsing metric files")
    # Initialize an empty list to store DataFrames
    dfs = []

    dtype_dict = {'base_address':int, 'length':int, 'adjusted_base':int, 'adjusted_end':int, 'size':float, 'label':str, 'dr_mean':float, 'dr_median':float, 'dr_max':float, 'ws_mean':float, 'ws_median':float, 'ws_max':float, 'd_mean':float, 'd_median':float, 'd_max':float, 'tr_mean':float, 'tr_median':float, 'tr_max':float}
    # Loop through each subdirectory in the main directory
    for subdir, _, files in os.walk(main_directory):
        # Loop through each file in the subdirectory
        for file in files:
            # Construct the full file path
            file_path = os.path.join(subdir, file)
            # Read the file content (assuming it's a string representation of a DataFrame)
            if metric_stats_type in file_path: 
                data_df = pd.read_csv(file_path)#, dtype=dtype_dict)
                app = str(file).split('_')[-1].split('.')[0]
                psize = float(str(file).split('_')[-2])
                
                if app not in ['spmv-coo-twitter7', 'GEMM'] and 'nopf' not in file and psize != 9.0: #Ignore spmv for now (const psize)
                # Add a new columns
                    data_df['app'] = app
                    data_df['psize'] = psize
                    data_df['sub_ratio'] = psize 
                    data_df['strategy'] = str('-' * len(data_df['label']))
                    #data_df['strategy'] = data_df['strategy'].astype(str)
                    data_df = data_df.sort_values('label').reset_index()
                    data_df['alloc'] = data_df.index
                    data_df['strat'] = list(solution[psize][app]) 
                    data_df['d'] = data_df['strat'] == 'd'
                    data_df['m'] = data_df['strat'] == 'm'
                    data_df['h'] = data_df['strat'] == 'h'
                    data_df['x'] = (app == "FDTD-2D") or (app == "stream")
                    #data_df['space_taken_by_placement'] = data_df['d'].astype(float) * data_df['size'] / 100.0
                    #data_df['space_taken_by_placement'] = data_df['space_taken_by_placement'].sum()
                    #data_df['space_remaining'] = 1 - data_df['space_taken_by_placement']
                    data_df['size'] = data_df['size'] / 100.0
                    data_df['anti_size'] = 1 - data_df['size']

                    data_df['ws_mean_1_OVER_size'] = data_df['d_mean_1'] * data_df['tr_mean_1']
                    #New cols of normal floats
                    #for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    #    data_df[f'{f}'] = f
                    if enable_RATIO:
                        for c in data_df.columns:
                            if "dr_" in c:
                                f = c.replace("dr_", "f_")
                                data_df[c] = data_df[c] / data_df[f]
                            if "ws_" in c:
                                f = c.replace("ws_", "f_")
                                data_df[c] = data_df[c] / data_df[f]

                    if enable_good_OR:
                        for metric1, metric2 in [('d_median_1','f_rel_mean_1'),('d_median_1000','f_rel_mean_1'),('d_mean_1000','f_rel_mean_1'),('d_mean_1', 'f_rel_mean_1'), ('dr_inter_max_1', 'ws_rel_mean_1000'), ('tr_median_1000', 'ts_rel_median_1000')]:
                            data_df[f'{metric1}_OR_{metric2}'] = data_df[[metric1, metric2]].max(axis=1)
                    if enable_OSOR:
                        #Add new metric, the max of two other metrics 
                        cols = data_df.columns
                        cols = set(cols) - set(['strat', 'd', 'h', 'm', 'x', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
                        cols = sorted([c for c in cols if numeric(data_df[c].iloc[0])])

                        tmp_df = pd.DataFrame(index=data_df.index)
                        for metric1 in ['dr_inter_mean_1000', 'ws_rel_max_1000']:
                            for j, metric2 in enumerate(cols):
                                tmp_df[f'{metric1}_OR_{metric2}'] = data_df[[metric1, metric2]].max(axis=1) 
                        data_df = pd.concat([data_df, tmp_df], axis=1).copy()

                    if enable_OR:
                        #Add new metric, the max of two other metrics 
                        cols = data_df.columns
                        cols = set(cols) - set(['strat', 'd', 'h', 'm', 'x', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
                        cols = sorted([c for c in cols if numeric(data_df[c].iloc[0])])

                        tmp_dfs = []
                        for i, metric1 in enumerate(cols):
                            tmp_df = pd.DataFrame(index=data_df.index)
                            for j, metric2 in enumerate(cols):
                                if j > i:
                                    tmp_df[f'{metric1}_OR_{metric2}'] = data_df[[metric1, metric2]].max(axis=1) 
                            tmp_dfs.append(tmp_df)
                        data_df = pd.concat([data_df] + tmp_dfs, axis=1).copy()
                   
                    #Add new metric combos to guess stream/fdtd 12 and 15
                    cols = data_df.columns
                    cols = set(cols) - set(['strat', 'd', 'h', 'm', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
                    cols = sorted([c for c in cols if numeric(data_df[c].iloc[0])])
                    tmp_dfs = [data_df]
                    stream_fdtd_metrics = []
                    for c in cols:
                        if "dr" in c:
                            stream_fdtd_metrics.append(c)
                    #dr_types = ['all', 'inter', 'intra']
                    #stream_fdtd_metrics = [f'dr_{t}_mean_1000' for t in dr_types] + [f'dr_{t}_mean_1' for t in dr_types]
                    
                    if ENABLE_DR_COMBOS:
                        for metric1 in cols:#cols:
                            for metric2 in stream_fdtd_metrics:
                                tmp_df = pd.DataFrame(index=data_df.index)
                                tmp_df[f'{metric2}_times_{metric1}'] = data_df[metric2] * data_df[metric1]
                                tmp_df[f'{metric2}_over_{metric1}'] = data_df[metric2] / data_df[metric1]
                                tmp_df[f'{metric2}_ORR_{metric1}'] = data_df[[metric1, metric2]].max(axis=1)
                                tmp_df[f'{metric2}_plus_{metric1}'] = data_df[metric2] + data_df[metric1]
                                tmp_df[f'{metric2}_minus_{metric1}'] = data_df[metric2] - data_df[metric1]

                                tmp_dfs.append(tmp_df)
                            for metric2 in ['size']:
                                tmp_df = pd.DataFrame(index=data_df.index)
                                tmp_df[f'{metric1}_times_{metric2}'] = data_df[metric1] * data_df[metric2]
                                tmp_df[f'{metric1}_over_{metric2}'] = data_df[metric1] / data_df[metric2]   

                                tmp_dfs.append(tmp_df)
                      
                        data_df = pd.concat(tmp_dfs, axis=1)

                    dfs.append(data_df)
                    print("File complete")

    # Concatenate all the DataFrames together
    combined_df = pd.concat(dfs,  ignore_index=False)
    numeric_df = combined_df.apply(pd.to_numeric, errors='coerce').fillna(combined_df)
    print("Finished parsing metric files")
    return numeric_df

def find_border_d_notd(data_df, metric):
    #Make dict of (m/h, metric value) pairs
    met_strat = {}
    for idx, row in data_df.iterrows():
        #Get the char of solution strategy corresponding to the row's allocation
        strat_str = solution[row['psize']][row['app']]
        strat_chr = strat_str[row['alloc']]
        met_strat[float(row[metric])] = strat_chr
        
    #Find max, min value of metric
    max_metric = max(list(met_strat.keys()))
    min_metric = min(list(met_strat.keys()))

    #Find best divider 
    scores = {}
    values = sorted(met_strat.keys())
    num_d = list(met_strat.values()).count('d')
    num_notd = len(met_strat.values()) - num_d

    #At starting divider, all d are correct, all notd are wrong
    score = num_d - num_notd
    scores[values[0]-1] = score
    best_score = score
    best_divider = values[0]-1
    d_is_up = True
    for v in values:
        if met_strat[v] == 'd':
            score-=1
        else:
            score+=1
        scores[v] = score
        if abs(score) > best_score:
            best_score = abs(score)
            best_divider = v
            d_is_up = (score > 0)
    
    #Return max score, best divider, which way is m 
    return best_score, best_divider, d_is_up

def find_range_x(data_df, metric):
    met_x = {}
    for idx, row in data_df.iterrows():
        met_x[float(row[metric])] = (float(row['x']) - 0.5) * 2 #Convert binary to -1/+1
        
    #Find max, min value of metric
    max_metric = max(met_x.keys())
    min_metric = min(met_x.keys())
    
    #Add buffer values that do not affect score
    met_x[max_metric+1] = 0
    met_x[min_metric-1] = 0

    #Find best range
    scores = {}
    values = sorted(met_x.keys())
    num_x = list(met_x.values()).count(1.0)
    num_notx = len(list(met_x.values())) - num_x

    #At starting divider, all x outside range
    score = num_x - num_notx 
    scores[values[0]-1] = score
    best_score = abs(score)
    best_divider = (values[0]-1, values[0] -1)
    x_inside = True
    improved = False
    
    prefix_sum = []
    score_list = []
    prefix_sum.append(0)
    for i, v in enumerate(values):
        score_list.append(met_x[v])
        prefix_sum.append(prefix_sum[-1] + score_list[-1])

    for i in range(len(values)):
        for j in range(i):
            score = sum(score_list[j:i]) - sum(score_list[:j]) - sum(score_list[i:])
            #score = prefix_sum[i+1] - prefix_sum[j+1]
            scores[(j, i)] = score
            if abs(score) > best_score:
                improved = True
                best_score = abs(score)
                best_dividers = (((values[j] + values[max(j-1, 0)]) / 2), (values[max(i-1, 0)] + values[min(i, len(values)-1)]) / 2) 
                x_inside = (score > 0)
    if improved: 
        #Return max score, best divider, which way is m 
        return metric, best_dividers, x_inside
    else:
        return metric, (0, 0), True


def find_border_x(data_df, metric):
    met_strat = {}
    for idx, row in data_df.iterrows():
        #Get the char of solution strategy corresponding to the row's allocation
        met_strat[float(row[metric])] = row['x']
        
    #Find max, min value of metric
    max_metric = max(list(met_strat.keys()))
    min_metric = min(list(met_strat.keys()))

    #Find best divider 
    scores = {}
    values = sorted(met_strat.keys())
    num_x = list(met_strat.values()).count('x')
    num_notx = len(list(met_strat.values())) - num_x

    #At starting divider, all m are correct, all h are wrong
    score = num_x - num_notx 
    scores[values[0]-1] = score
    best_score = score
    best_divider = values[0]-1
    x_is_up = True
    improved = False
    for i, v in enumerate(values):
        if met_strat[v]:
            score-=1
        else:
            score+=1
        scores[v] = score
        if abs(score) > best_score and i < len(values) - 1:
            improved = True
            best_score = abs(score)
            best_divider = (values[i] + values[i+1]) / 2
            x_is_up = (score > 0)
    if improved: 
        #Return max score, best divider, which way is m 
        return metric, best_divider, x_is_up
    else:
        return metric, 0, True



def find_border_h_m(data_df, metric):
    #data_df only contains allocs with h or m 
    #Make dict of (m/h, metric value) pairs
    met_strat = {}
    for idx, row in data_df.iterrows():
        #Get the char of solution strategy corresponding to the row's allocation
        met_strat[float(row[metric])] = row['strat']
        
    #Find max, min value of metric
    max_metric = max(list(met_strat.keys()))
    min_metric = min(list(met_strat.keys()))

    #Find best divider 
    scores = {}
    values = sorted(met_strat.keys())
    num_m = list(met_strat.values()).count('m')
    num_h = list(met_strat.values()).count('h')

    #At starting divider, all m are correct, all h are wrong
    score = num_m - num_h 
    scores[values[0]-1] = score
    best_score = score
    best_divider = values[0]-1
    m_is_up = True
    improved = False
    for i, v in enumerate(values):
        if met_strat[v] == 'm':
            score-=1
        elif met_strat[v] == 'h':
            score+=1
        scores[v] = score
        if abs(score) > best_score and i < len(values) - 1:
            improved = True
            best_score = abs(score)
            best_divider = (values[i] + values[i+1]) / 2
            m_is_up = (score > 0)
    if improved: 
        #Return max score, best divider, which way is m 
        return best_score, best_divider, m_is_up
    else:
        return 0, 0, True

def numeric(s):
    try:
        float(s)
        return True
    except TypeError:
        return False

    except ValueError:
        return False
def rank_d_notd_classifiers(data_df):
    scores_to_dividers = {}
    metrics_to_tuples = {}
    for metric in data_df.columns:
        if numeric(data_df[metric].iloc[0]):
            best_score, best_divider, d_is_up = find_border_d_notd(data_df, metric)
            metrics_to_tuples[metric] = (metric, best_divider, d_is_up)
            if best_score in scores_to_dividers:
                scores_to_dividers[best_score].append((metric, best_divider, d_is_up))
            else:
                scores_to_dividers[best_score] = [(metric, best_divider, d_is_up)]
    max_score = max(list(scores_to_dividers.keys()))
    print(f"Max score is {max_score}\nThese metrics achieve it:")
    print(scores_to_dividers[max_score])
    print("These were close:")
    print("Off by 1:\n", scores_to_dividers[max_score - 1])
    print("Off by 2:\n", scores_to_dividers[max_score - 2])
    columns = []
    classifiers = []
    data_df['d_vote'] = 0
    for leeway in [0, 1, 2]:
        for metric, divider, d_is_up in scores_to_dividers[max_score - leeway]:
            columns.append(f'{metric}_d')
            classifiers.append((metric, divider, d_is_up))
            data_df[f'{metric}_d'] = False
            if d_is_up:
                data_df.loc[data_df[metric] >= divider, f'{metric}_d'] = True
            else: 
                data_df.loc[data_df[metric] < divider, f'{metric}_d'] = True
            data_df['d_vote'] = data_df['d_vote'] + data_df[f'{metric}_d']
    #data_df['d_vote'] = data_df['d_vote'] > len(columns) / 2
    data_df = data_df.sort_values('d')
    print(data_df[columns + ['d_vote','d']])
    return metrics_to_tuples, classifiers



def rank_m_h_classifiers(data_df, p=False):
    data_df = data_df[data_df['d'] == False]
    cols = data_df.columns
    #cols = [c for c in cols if 'OR' not in c]
    scores_to_dividers = {}
    metrics_to_tuples = {}
    for metric in data_df.columns:
        if metric in cols and len(data_df[metric]) > 0 and numeric(data_df[metric].iloc[0]):
            best_score, best_divider, m_is_up = find_border_h_m(data_df, metric)
            metrics_to_tuples[metric] = (metric, best_divider, m_is_up)
            if best_score in scores_to_dividers:
                scores_to_dividers[best_score].append((metric, best_divider, m_is_up))
            else:
                scores_to_dividers[best_score] = [(metric, best_divider, m_is_up)]
    max_score = -9999999
    if len(scores_to_dividers.keys()) > 0:
        max_score = max(list(scores_to_dividers.keys()))
    else: 
        print("Error in rank_m_h_classifiers")
        exit()
    if(p):
        print(f"Max score is {max_score}\nThese metrics achieve it:")
        print(scores_to_dividers[max_score])
        print("These were close:")
        if max_score - 1 in scores_to_dividers:
            print("Off by 1:\n", scores_to_dividers[max_score - 1])
        if max_score - 2 in scores_to_dividers:
            print("Off by 2:\n", scores_to_dividers[max_score - 2])
    columns = []
    classifiers = []
    data_df['vote'] = 0
    for leeway in [0, 1, 2]:
        if max_score - leeway in scores_to_dividers:
            for metric, divider, m_is_up in scores_to_dividers[max_score - leeway]:
                columns.append(f'{metric}_m')
                classifiers.append((metric, divider, m_is_up))
                data_df[f'{metric}_m'] = False
                if m_is_up:
                    data_df.loc[data_df[metric] >= divider, f'{metric}_m'] = True
                else: 
                    data_df.loc[data_df[metric] < divider, f'{metric}_m'] = True
                data_df['vote'] = data_df['vote'] + data_df[f'{metric}_m']
    data_df['vote'] = data_df['vote'] > len(columns) / 2
    data_df = data_df.sort_values('m')
    data_df = data_df.copy()
    if p: 
        print(data_df[columns + ['vote','m']])
    return metrics_to_tuples, classifiers

def score_x(data_df):
    score = 0.5 * (data_df['predict_x'] & data_df['x']).sum()
    score += 0.5 * (data_df['predict_x'] == data_df['x']).sum()
    return score

def predict_x(data_df, metric_tuple, p=False):
    metric, thold, direction = metric_tuple
    if metric not in data_df.columns:
        print(f"Error: {metric} is not a column of df")
    if isinstance(thold, tuple):
        bottom, top = thold
        x_inside = direction
        if x_inside:
            data_df['predict_x'] = (data_df[metric] <= top) & (data_df[metric] >= bottom)
        else:
            data_df['predict_x'] = (data_df[metric] > top) | (data_df[metric] < bottom)

    else:
        x_is_up = direction
        if x_is_up:
            data_df['predict_x'] = data_df[metric] >= thold
        else:
            data_df['predict_x'] = data_df[metric] < thold 
    
    score = score_x(data_df)
    if(p):
        print(f"Thold: {thold}")
        print(f"X Up/Inside: {direction}")
        print_x_classification(data_df, metric)
    return score

def score_row_m_classifier(data_df):
    data_df = data_df[data_df['d']==False]
    predict_m(data_df)
    data_df['predict_m_score'] = data_df['predict_m'] & data_df['m']
    score = data_df['predict_m_score'].astype(int).sum()
    print(score, len(data_df))

def classify_d_hard_code(data_df, p=False, apply_m=False, perf_df=pd.DataFrame()):
    
    #Different classifier for [FDTD-2D, stream] and others
    data_df['new'] = data_df['tr_mean_1000_OR_tr_prop_0s_1']
    fdtd_stream_view = data_df[data_df['app'].isin(['FDTD-2D', 'stream'])]
    data_df.loc[fdtd_stream_view.index, 'new'] = fdtd_stream_view['d_prop_max_1000_OR_size'] * -1
   
    #For knapsack, all values should be positive
    min_item = data_df["new"].min()
    if min_item < 0: 
        data_df["new"] = data_df["new"] + (1.001 * abs(float(min_item)))
    
    
    min_item = data_df["new"].min()
    if min_item <= 0: 
        data_df["new"] = data_df["new"] + 0.001
    
    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])
    # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='strategy').sort_values(by='app', key=lambda x: x.str.lower())
    #pivot_table = solution.copy()
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group.sort_values('label').reset_index(drop=True)
        strategy = ['-'] * len(tmp)
        # Heuristic-ify the strategy
        capacity = .99
        chosen = knapsack(tmp, capacity, 'size', 'new')
        if len(chosen) == 0:
            print(f"Bad Knapsack for metric {classifier}")
            return -1000
        for loc in chosen:
            strategy[loc] = 'd'
            capacity -= tmp['size'][loc]

        if apply_m:
            for loc, c in enumerate(strategy):
                if c == '-':
                    if tmp['m_prediction'][loc]:
                        strategy[loc] = 'm'
                    else:
                        strategy[loc] = 'h'

        pivot_table.loc[idx[0], idx[1]] = str(''.join(strategy))

    comparison_result, score = compare_pivot_tables(pivot_table, solution)
    comparison_result2, comparison_default, score2 = compare_pivot_tables_speedup(pivot_table, perf_df.copy())
    # Print the pivot table
    if(p):
        print(pivot_table)
        print(solution)
        print('-' * 15)
        print(comparison_result)
        print(f"Score: {score}/20")
        if not perf_df.empty:
            comparison_result, comparison_default, score2 = compare_pivot_tables_speedup(pivot_table, perf_df.copy())
            print('-' * 15)
            print('Perf diff vs optimal')
            print(comparison_result)
            print(f"Score (golf rules): {score2}")
            print('-' * 15)
            print('Speedup vs default')
            print(comparison_default)
    return -1 * score2[0]

def classify_d_with(data_df, classifier, scalar, p=False, apply_m=False, perf_df=pd.DataFrame(), returnTable=False):
    skip_knapsack = False
    if classifier == 'none':
        skip_knapsack = True

    # Initialize an empty list to store DataFrames
    else:
        if scalar == 1:
            data_df["new"] = data_df[classifier]
        if scalar == -1:
            data_df["new"] = data_df[classifier] * -1
        
        #For knapsack, all values should be positive
        min_item = data_df["new"].min()
        if min_item < 0: 
            data_df["new"] = data_df["new"] + (1.001 * abs(float(min_item)))
        if min_item <= 0: 
            data_df["new"] = data_df["new"] + 0.001
    
    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])
    # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='strategy').sort_values(by='app', key=lambda x: x.str.lower())
    #pivot_table = solution.copy()
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group.sort_values('label').reset_index(drop=True)
        strategy = ['-'] * len(tmp)
        # Heuristic-ify the strategy
        #If threshold is high, use t
        #d first
        capacity = .99
        chosen = []
        if not skip_knapsack: 
            chosen = knapsack(tmp, capacity, 'size', 'new')
            if len(chosen) == 0 and min(tmp['size']) < capacity:
                print(f"Bad Knapsack for metric {classifier}")
                return -1000
            for loc in chosen:
                strategy[loc] = 'd'

        if apply_m:
            for loc, c in enumerate(strategy):
                if c == '-':
                    if tmp['m_prediction'][loc]:
                        strategy[loc] = 'm'
                    else:
                        strategy[loc] = 'h'

        pivot_table.loc[idx[0], idx[1]] = str(''.join(strategy))

    comparison_result2, comparison_default, score2 = compare_pivot_tables_speedup(pivot_table, perf_df.copy())
    if(p):
        comparison_result, score = compare_pivot_tables(pivot_table, solution)
        print(pivot_table)
        print(solution)
        print('-' * 15)
        print(comparison_result)
        print(f"Score: {score}/20")
        #if not perf_df.empty:
        #    comparison_result, comparison_default, score2 = compare_pivot_tables_speedup(pivot_table, perf_df.copy())
        print('-' * 15)
        print('Perf diff vs optimal')
        print(comparison_result2)
        print(f"Score (golf rules): {score2}")
        print('-' * 15)
        print('Speedup vs default')
        print(comparison_default)
    if returnTable:
        return comparison_default
    return -1 * score2[0]


def filter_corr_matrix_by_threshold(df, threshold, target):
    """
    Filters the correlation matrix to include only columns that have a correlation
    with column_name greater than the threshold or less than -threshold.
    
    Parameters:
    - df: pandas DataFrame (original data)
    - corr_matrix: pandas DataFrame (correlation matrix)
    - threshold: float (correlation threshold)
    - column_name: str (the column to filter by)
    
    Returns:
    - filtered_corr_matrix: pandas DataFrame (filtered correlation matrix)
    """
    # Initialize a list to store columns that meet the threshold
    selected_columns = []

    # Loop through each column except 'm' and calculate correlation with 'm'
    #print("Selecing columsn with useful chi values")
    for col in df.columns.drop(target):
        if df[col].nunique() != 1:
            #correlation = df[col].corr(df[target])
            correlation = corr(df.copy(), col, target) #Chi beats corr for binary vars
            if abs(correlation) > threshold:
                selected_columns.append(col)

    # Add 'm' to the list of selected columns to keep it in the result
    selected_columns.append(target)

    # Filter the DataFrame to include only selected columns
    num = len(selected_columns)
    #print(f"Calculating Chi Matrix from {num} selections")
    df_selected = df[selected_columns]
    return df_corr(df_selected)



def get_corr_matrix(data_df, p=False):
    #Drop non-numeric columns
    for column in data_df.columns:
        if not numeric(data_df[column].iloc[0]):
            data_df = data_df.drop(columns=[column])
        elif data_df[column].nunique == 1 and column not in ['x', 'd', 'm', 'h']:
            data_df = data_df.drop(columns=[column])

    #Drop irrelevant columns
    data_df = data_df.drop(columns=['level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
    
    #Refine parameters
    corr_matrix_x = []
    if data_df['x'].nunique != 1:
        corr_matrix_x = filter_corr_matrix_by_threshold(data_df.copy(), 0.10, 'x')
        #corr_matrix_d = corr_matrix_d.to_pandas()
        if(p):
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix_d, annot=True, cmap='coolwarm', cbar=True)
            plt.savefig('/home/najones/uvm-eviction/figs/{hostname}/correlation_matrix_x.png')


    #Refine parameters
    corr_matrix_d = []
    if data_df['d'].nunique != 1:
        corr_matrix_d = filter_corr_matrix_by_threshold(data_df.copy(), 0.10, 'd')
        #corr_matrix_d = corr_matrix_d.to_pandas()
        if(p):
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix_d, annot=True, cmap='coolwarm', cbar=True)
            plt.savefig('/home/najones/uvm-eviction/figs/{hostname}/correlation_matrix_d.png')

    #Now do the h vs m columns, after d is eliminated
    data_df = data_df[data_df['d'] == False]
    cols = [c for c in data_df.columns if 'OR' not in c and data_df[c].nunique != 1]
    corr_matrix_hm = []
    if data_df['m'].nunique() != 1:
        data_df = data_df[cols]
        corr_matrix_hm = filter_corr_matrix_by_threshold(data_df.copy(), 0.30, 'm')
        #corr_matrix_hm = corr_matrix_hm.to_pandas()
        if(p):
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix_hm, annot=True, cmap='coolwarm', cbar=True)
            plt.savefig('/home/najones/uvm-eviction/figs/{hostname}/correlation_matrix_hm.png')
    return corr_matrix_d, corr_matrix_hm, corr_matrix_x

def select_columns_with_strong_relationship(corr_matrix, column_name, max_cols=5, threshold=0.9):
    """
    Selects columns that have the strongest correlation with `column_name` and ensures 
    that none of the selected columns have abs(correlation) > `threshold` with each other.
    
    Parameters:
    - corr_matrix: pandas DataFrame (correlation matrix)
    - column_name: str (the column to compare correlations against)
    - max_cols: int (maximum number of columns to select)
    - threshold: float (absolute correlation threshold between selected columns)
    
    Returns:
    - selected_columns: list (columns with the strongest correlation to `column_name`)
    """
    if len(corr_matrix) == 0:
        return [column_name]
    # Step 1: Get absolute correlations with the target column
    target_corr = corr_matrix[column_name].abs().sort_values(ascending=False)
    
    # Step 2: Drop the target column itself from the list
    target_corr = target_corr.drop(index=column_name)
    
    # Step 3: Select columns iteratively, ensuring no high correlation among them
    selected_columns = []
    blacklist = ['d', 'h', 'm', 'x']
    
    for col in target_corr.index:
        if len(selected_columns) >= max_cols:
            break
        
        # Check if the new column has a high correlation with any already selected column
        is_highly_correlated = any(corr_matrix[col].abs()[selected_col] > threshold for selected_col in selected_columns)
        
        if not is_highly_correlated and col not in blacklist:
            selected_columns.append(col)
    
    return selected_columns


def get_smallest_corr_matrix(data_df):
    
    corr_d, corr_m, corr_x = get_corr_matrix(data_df.copy())
    best_d_metrics = select_columns_with_strong_relationship(corr_d, 'd', max_cols=3, threshold=0.99)
    best_m_metrics = select_columns_with_strong_relationship(corr_m, 'm', max_cols=9, threshold=0.9)
    best_metrics = best_d_metrics + best_m_metrics
    datad_df = data_df[best_metrics + ['d']]
    datahm_df = data_df[data_df['d']==False]
    datahm_df = datahm_df[best_metrics + ['h', 'm']]
    corr_matrix_d = datad_df.corr()
    corr_matrix_hm = datahm_df.corr()
    

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix_d, annot=True, cmap='coolwarm', cbar=True)
    plt.savefig('/home/najones/uvm-eviction/figs/{hostname}/small_correlation_matrix_metrics_d.png')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix_hm, annot=True, cmap='coolwarm', cbar=True)
    plt.savefig('/home/najones/uvm-eviction/figs/{hostname}/small_correlation_matrix_metrics_hm.png')

def brute_force_metrics_vs_d(data_df, metrics=[], perf_df=pd.DataFrame(), p=False):
    #Drop unrelated, nonnumeric, and/or constant columns
    df = data_df.drop(columns=['strat', 'd', 'h', 'm', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc', 'size', 'anti_size', 'length', 'psize', 'sub_ratio'])
    df = df.select_dtypes(include=['number'])
    df = df.loc[:, (df.nunique() > 1)]
    
    if len(metrics) > 0:
        df = df[['app', 'psize'] + metrics]
    class_dict = {}
    metrics_to_tuples = {}
    classifiers = df.columns
    max_score = -10000000
    best = ""
    percent = 0
    for i, classifier in enumerate(classifiers):
        if len(classifiers) > 100 and i % (len(classifiers) // 100) == 0:
            percent += 1
            print(f'{percent}% Complete with {len(classifiers)} classifiers')
        for m in [-1, 1]:
            invert = False
            mode = "knapsack"
            score = classify_d_with(data_df, classifier, m, p=False, apply_m=False, perf_df=perf_df)
            metric_tuple = (classifier, m)
            if classifier not in metrics_to_tuples:
                metrics_to_tuples[classifier] = (metric_tuple, score)
            elif score > metrics_to_tuples[classifier][1]:
                metrics_to_tuples[classifier] = (metric_tuple, score)
            class_dict[(classifier,m)] = score
            if score > max_score:
                max_score = score
                best = (classifier, m)

    print(f"Best Score: {max_score}")
    #print("classifier dict", class_dict)
    best_mets = []
    if p:
        print('-' * 30)
    for key, val in class_dict.items():
        if val >= max_score -2:
            if p:
                print(key, val)
        if val == max_score:
            best_mets.append(key)
            #best = key
            #classify_d_with(df.copy(), best[0], best[1], best[2], True, best[3])
    for metric in best_mets:
        if p:
            print('-' * 30)
            print("best", metric)
            print(f"Classifying with {metric}")
            classify_d_with(data_df.copy(), metric[0], metric[1], p=True, apply_m=False, perf_df=perf_df.copy())
    #score = classify_with_trio(df.copy())
    #print(score)
    return metrics_to_tuples, best_mets[0], best_mets

def set_d_prediction(data_df, d_predictors):
    data_df['d_vote'] = 0
    columns = []
    for metric, divider, d_is_up in d_predictors:
        data_df[f'{metric}_d'] = False
        columns.append(f'{metric}_d')
        if d_is_up:
            data_df.loc[data_df[metric] >= divider, f'{metric}_d'] = True
        else: 
            data_df.loc[data_df[metric] < divider, f'{metric}_d'] = True
        data_df['d_vote'] = data_df['d_vote'] + data_df[f'{metric}_d']
    data_df['d_prediction'] = data_df['d_vote'] > len(d_predictors) / 2
    data_df_view = data_df.sort_values('d')
    print(data_df_view[columns + ['d_prediction','d']])
    return ('d_vote', 1, 'knapsack', False, False)

def brute_force_find_ensemble(data_df, d_predictor, m_tuples, df_perf):
    # Loop over all subsets
    max_score = -10000
    best_ensemble = m_tuples
    for r in range(len(m_tuples) + 1):
        for subset in combinations(m_tuples, r):
            tmp = data_df.copy()
            set_m_prediction(tmp, subset, p=False)
            score = classify_d_with(tmp, d_predictor[0], d_predictor[1], d_predictor[2], False, invert=d_predictor[3], apply_m=True, perf_df=df_perf)
            if score > max_score:
                best_ensemble = subset
    return best_ensemble
            

def write_metrics(data_df, d_tuple, m_tuples):
    #m_metrics
    data_df['m_votes'] = 0
    columns = ['app', 'label']
    for metric, divider, m_is_up in m_tuples:
        data_df[f'{metric}_m'] = False
        data_df[f'{metric}_thold'] = divider
        data_df[f'{metric}_mup'] = m_is_up
        columns = columns +[metric, f'{metric}_m', f'{metric}_thold', f'{metric}_mup']
        if m_is_up:
            data_df.loc[data_df[metric] >= divider, f'{metric}_m'] = True
        else: 
            data_df.loc[data_df[metric] < divider, f'{metric}_m'] = True
        data_df['m_votes'] = data_df['m_votes'] + data_df[f'{metric}_m']
    data_df['m_prediction'] = data_df['m_votes'] > len(m_tuples) / 2
    columns += ['m_votes', 'm_prediction', 'm']

    #d_metrics 
    data_df[f'{d_tuple[0]}_d'] = data_df[d_tuple[0]]
    columns += [d_tuple[0], f'{d_tuple[0]}_d', 'd']
    if d_tuple[1] == -1:
        data_df[f'{d_tuple[0]}_d'] = data_df[f'{d_tuple[0]}_d'] * -1
    
    #For knapsack, all values should be positive
    min_item = data_df[f'{d_tuple[0]}_d'].min()
    if min_item < 0: 
        data_df[f'{d_tuple[0]}_d'] = data_df[f'{d_tuple[0]}_d'] + (1.001 * abs(float(min_item)))
     
    min_item = data_df[f'{d_tuple[0]}_d'].min()
    if min_item <= 0: 
        data_df[f'{d_tuple[0]}_d'] = data_df[f'{d_tuple[0]}_d'] + 0.001
    
    
    #Write to File
    data_df_view = data_df[columns]
    data_df_view.to_csv('metrics.csv', index=False)


def set_m_prediction(data_df, m_predictors, p=False):
    if 'm' in m_predictors:
        data_df['m_prediction'] = data_df['m']
        #print("Using oracle m prediction")
        return
    if 'all' in m_predictors:
        data_df['m_prediction'] = True
        #print("Using 'all' m prediction")
        return
    if 'none' in m_predictors:
        data_df['m_prediction'] = False
        #print("Using 'none' m prediction")
        return

    data_df['vote'] = 0
    columns = []
    for metric, divider, m_is_up in m_predictors:
        data_df[f'{metric}_m'] = False
        columns.append(f'{metric}_m')
        if m_is_up:
            data_df.loc[data_df[metric] >= divider, f'{metric}_m'] = True
        else: 
            data_df.loc[data_df[metric] < divider, f'{metric}_m'] = True
        data_df['vote'] = data_df['vote'] + data_df[f'{metric}_m']
    data_df['m_prediction'] = data_df['vote'] > len(m_predictors) / 2
    data_df_view = data_df[data_df['d'] == False].sort_values('m')
    if(p):
        print(data_df_view[columns + ['m_prediction','m']])
        data_df.to_csv('metrics.csv', index=False)

def classify_m_given_d(df_data, df_perf, m_predictors, p=False):
    set_m_prediction(df_data, m_predictors, p)
    return classify_d_hard_code(df_data, p=p, apply_m=True, perf_df=df_perf)

def classify_both(df_data, df_perf, d_predictor, m_predictors, p=False, returnTable=False):
    set_m_prediction(df_data, m_predictors, p)
    return classify_d_with(df_data, d_predictor[0], d_predictor[1], p=p, apply_m=True, perf_df=df_perf, returnTable=returnTable)

def vote_for_both(df_data, df_perf, d_predictors, m_predictors):
    set_m_prediction(df_data, m_predictors)
    d_predictor = set_d_prediction(df_data, d_predictors)
    classify_d_with(df_data, d_predictor[0], d_predictor[1], d_predictor[2], True, invert=d_predictor[3], apply_m=True, perf_df=df_perf)

def train_classify_app_group(df, applist):
    df['x'] = 0
    marked_view = df[df['app'].isin(applist)]
    df.loc[marked_view.index, 'x'] = 1

    blacklist = ['x', 'strat', 'd', 'h', 'm', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc']
    xmets = [find_range_x(df.copy(), met) for met in df.columns if met not in blacklist and numeric(df[met].iloc[0])]
    scores_to_xmets = {}
    max_score = 0
    for met in xmets:
        score = predict_x(df.copy(), met)
        if score not in scores_to_xmets:
            scores_to_xmets[score] = [met]
        else:
            scores_to_xmets[score].append(met)
        if score > max_score:
            max_score = score
    if max_score == 0:
        print("Unable to predict.")
    else:
        print(f"Max score is {max_score}")
        for i, met in enumerate(scores_to_xmets[max_score]):
            print(f"Met: {met}")
            print(f"Real range: {(df[met[0]].min(), df[met[0]].max())}")
            predict_x(df.copy(), met, True)
            if (i == 100):
                print(f"And {len(scores_to_xmets[max_score]) - 4} others")
                return

def score_df(df, pf, p=False):
    score = 0
    corr_d, corr_m, corr_x = get_corr_matrix(df.copy().fillna(-1))
    d_dict, brute_force_best, best_ds = brute_force_metrics_vs_d(df.copy(), metrics=[], perf_df=pf.copy(), p=p)
    best_d_tuple = brute_force_best
    score += classify_both(df.copy(), pf.copy(), best_d_tuple, ['m'], p)
    for d in sorted(best_ds, key=lambda x: x[0]):
        print(d)
    #write_metrics(df.copy(), best_d_tuple, best_m_tuples)
    return best_ds



def score_mutex_dfs(df1, df2, pf, p=False):
    score = 0
    for df in [df1.copy(), df2.copy()]:
        score += score_df(df, pf, p)
    return score

def app_is_streaming(app_df):
    #If all (significant) allocations have dr_intra_mean_1 > 0.65
    #AND
    #all (significant) allocations have tr_mean_1 < 0.005
    dr_condition = app_df['dr_intra_mean_1000'].between(1, 1000, inclusive="both").all()
    tr_condition = app_df['tr_mean_1'].between(0, 0.005, inclusive="both").all()
    return dr_condition and tr_condition


def classify_apps_hardcode(data_df):
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group #[group['label'] != 'all']
        x_class = False 
        if app_is_streaming(tmp):
            x_class = True 
        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'x_class'] = x_class

def print_metric(data_df, metric):
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group[group['label'] == 'all']
        classification = f'{round(tmp[metric].iloc[0], 2)}'
        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'x_class'] = str(classification)

    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])                                                              # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='x_class').sort_values(by='app', key=lambda x: x.str.lower())
    print(pivot_table)

def print_metric_abc(data_df, metric):
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        classification = "".join([f'({round(a, 2)})' for a in group[metric]])
        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'x_class'] = str(classification)

    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])                                                              # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='x_class').sort_values(by='app', key=lambda x: x.str.lower())
    print(pivot_table)

    

def print_apps_hardcode(data_df):
    data_df['x_class'] = ""
    groups = data_df.groupby(['app', 'psize'])
    for idx, group in groups:
        tmp = group[group['label'] != 'all']
        #tmp = group.groupby(['label']).agg({'dr_mean_1':'mean', 'label':'first', 'app':'first', 'psize':'first'})
        #classification = ''.join(f'{round(a, 2), }' for a in group['dr_mean_1'])
        classification = '-'#f'{round(tmp["dr_mean_1"].iloc[0], 2)}'
        # Heuristic-ify the strategy
        #If threshold is high, use t
        #d first
        if app_is_streaming(tmp):
            classification = 'x'
        #if app_is_fdtd2d(tmp):
        #    classification = 'f'
        #if app_is_stream(tmp):
        #if True:
        #    classification = 's'
            #classification = f"{tmp[metric].iloc[0]}"

        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'x_class'] = str(classification)

    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])                                                              # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='x_class').sort_values(by='app', key=lambda x: x.str.lower())
    print(pivot_table)

def printall(a):
    for b in a: 
        print(b)
   
def predict_unseen_strat(data_df):
    classify_apps_hardcode(data_df)
    
    df1 = data_df[data_df['x_class'] == True]
    m_metric = ('d_mean_1000', 0.15, True)
    d_metric = ('dc_intra_rel_mean_1', -1)
    set_m_prediction(df1, [m_metric])
    classify_d_with(df1, d_predictor[0], d_predictor[1], p=True, apply_m=True)
 
    df2 = data_df[data_df['x_class'] == False].copy()
    bc = 1000
    print_metric_abc(df2, f'd_mean_{bc}')
    print_metric_abc(df2, f'dr_intra_mean_{bc}')
    #Set m_prediction 
    df2['thold1'] = 0.02
    df2['thold2']= 1
    df2['cond1'] = df2[f'd_mean_{bc}'] > df2['thold1']
    df2['cond2'] = df2[f'dr_intra_mean_{bc}'] > df2['thold2']
    df2['m_prediction'] = df2['cond1'] & df2['cond2']
    d_metric = (f'tr_median_{bc}_OR_ts_rel_median_{bc}', 1)
    classify_d_with(df2.copy(), d_metric[0], d_metric[1], p=True, apply_m=True, perf_df=pf)

def predict_strat_hardcode(data_df, pf, p=True):
    classify_apps_hardcode(data_df)
    
    df1 = data_df[data_df['x_class'] == True]
    #print_metric_abc(df1.copy(), 'd_mean_1000')
    m_metric = ('d_mean_1000', 0.13, True)
    d_metric = ('dc_intra_rel_mean_1000', -1)
    table1 = classify_both(df1.copy(), pf.copy(), d_metric, [m_metric], p=p, returnTable=True)
 
    df2 = data_df[data_df['x_class'] == False].copy()
    bc = 1000
    #Set m_prediction 
    df2['thold1'] = 0.02
    df2['thold2']= 1
    df2['cond1'] = df2[f'd_mean_{bc}'] > df2['thold1']
    df2['cond2'] = df2[f'dr_intra_mean_{bc}'] > df2['thold2']
    df2['m_prediction'] = df2['cond1'] & df2['cond2']
    d_metric = (f'tr_median_{bc}_OR_ts_rel_median_{bc}', 1)
    table2 = classify_d_with(df2.copy(), d_metric[0], d_metric[1], p=p, apply_m=True, perf_df=pf, returnTable=True)
    
    all_vs_default = pd.concat([table1, table2]).reset_index(drop=False)
    #print(all_vs_default)
    return all_vs_default

def print_paper_mets(data_df):
    data_df = data_df[PAPER_METS]
    data_df.to_csv("paper_metrics.csv", index=False)

def predict_strat_naive(data_df, pf):
    d_metric = ('ws_mean_1_OVER_size', 1)
    classify_both(data_df.copy(), pf.copy(), d_metric, ['m'], p=True)

def reshape_pivot_tables(pivot_tables, titles):
    """
    Reshape a list of pivot tables into a single DataFrame.
    
    Parameters:
    pivot_tables (list of pd.DataFrame): List of pivot tables with psize as columns and app as index.
    titles (list of str): Titles corresponding to each pivot table.
    
    Returns:
    pd.DataFrame: Reshaped DataFrame with columns indexed by app and psize.
    """
    reshaped_dfs = []
    
    for table, title in zip(pivot_tables, titles):
        df_melted = table.reset_index().melt(id_vars='app', var_name='psize', value_name=title)
        reshaped_dfs.append(df_melted)
    
    result = reshaped_dfs[0]
    for df in reshaped_dfs[1:]:
        result = result.merge(df, on=['app', 'psize'], how='outer')
    
    return result



def print_strat_naive(data_df, pf, p=False):
    pivots = []
    strat_names = ['Migr.','Host','Dev+Migr.','Dev+Host','MIM','Opt']
    #All m or All h
    d_metric = ('none', 1)
    pivots.append(classify_both(data_df.copy(), pf.copy(), d_metric, ['all'], p=p, returnTable=True))
    pivots.append(classify_both(data_df.copy(), pf.copy(), d_metric, ['none'], p=p, returnTable=True))


    #d using unique faults / alloc size, then all m or all h
    d_metric = ('ws_mean_1_OVER_size', 1)
    pivots.append(classify_both(data_df.copy(), pf.copy(), d_metric, ['all'], p=p, returnTable=True))
    pivots.append(classify_both(data_df.copy(), pf.copy(), d_metric, ['none'], p=p, returnTable=True))

    #Our prediction 
    pivots.append(predict_strat_hardcode(data_df.copy(), pf.copy(), p=p))

    #Optimal
    a, comparison_default, b = compare_pivot_tables_speedup(solution, pf.copy())
    pivots.append(comparison_default)

    df = reshape_pivot_tables(pivots, strat_names)
    df = df[df['psize'] != 'index']
    df.to_csv("paper_metrics.csv", index=False)

def make_all_perf_barchart(csv_file):
    textsize = 20 
    plt.rcParams.update({'font.size': textsize})
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Convert app names to lowercase
    df["app"] = df["app"].str.lower()

    # Compute geometric mean for each app
    df_geomean = df.groupby("app").agg(lambda x: np.exp(np.log(x).mean())).reset_index()

    # Ensure 'cg' and 'tealeaf' are the last clusters
    special_apps = ["cg", "tealeaf"]
    df_geomean = pd.concat([
        df_geomean[~df_geomean["app"].isin(special_apps)],
        df_geomean[df_geomean["app"].isin(special_apps)]
    ]).reset_index(drop=True)
    
    # Set up the plot
    metrics = ["Migr.", "Host", "Dev+Migr.", "Dev+Host", "MIM"]
    #colors = ["blue", "orange", "green", "red", "purple"]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
    hatch_patterns = ['/', '\\', 'x', '-', 'o']
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width and positions
    bar_width = 0.15
    x = np.arange(len(df_geomean))

    max_height = 6
    min_height = 0.25
    # Plot each metric as a separate bar in the cluster
    for i, metric in enumerate(metrics):
        bar = ax.bar(x + i * bar_width, df_geomean[metric].apply(lambda x: max(x, min_height + 0.01)), width=bar_width, label=metric, color=colors[i], hatch=hatch_patterns[i])
        #bar.set_hatch(hatches[i])
    # Annotate each too-tall bar with its value
    for p in ax.patches:
        if p.get_height() > max_height:
            ha='center'
            if p.get_height() > 40:
                ha='left'
            ax.annotate(str(int(np.round(p.get_height(), 0))), 
                    (p.get_x() + p.get_width() / 2., max_height),
                    ha=ha, va='bottom', fontsize=18, rotation=35)

    # Add a dotted vertical line before 'cg' and 'tealeaf'
    split_idx = 4.80
    ax.axvline(split_idx, linestyle="dotted", color="black", linewidth=1.5)

    # Format the plot
    ax.set_xticks(x + (len(metrics) - 1) * bar_width / 2)
    ax.set_xticklabels(df_geomean["app"], rotation=15, ha="right")
    ax.legend(title="Strategy", fontsize=18, title_fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5, columnspacing=0.5, handletextpad=0.25)

    plt.hlines(1, -0.05,len(df_geomean)-0.35)
    plt.ylabel("Speedup (vs Migr.)")
    plt.yscale('log', base=2)  # Set y-axis to logarithmic
    plt.yticks([0.25, 0.5, 1, 2, 4])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(min_height, max_height)
    from matplotlib.ticker import ScalarFormatter
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    
    # Show the plot
    os.makedirs(f"../../figs/{hostname}/perf_comparison/", exist_ok=True)
    plt.savefig(f'../../figs/{hostname}/perf_comparison/default_{kernel_version}_vanilla_geomean_each.png')
    plt.close()


def make_perf_barchart(csv_file):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, header=0)
    textsize = 20 
    plt.rcParams.update({'font.size': textsize})
    # Ensure required columns exist
    if 'app' not in df or 'psize' not in df:
        raise ValueError("CSV must contain 'app' and 'psize' columns.")

    # Convert numeric columns to float
    numeric_cols = df.columns.difference(['app', 'psize'])
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['Static-Opt'] = df[['Migr.', 'Host', 'Dev+Migr.', 'Dev+Host']].max(axis=1)

    for idx, row in df.iterrows():
        # Extract title information
        title = f"{row.loc['app']} - {row.loc['psize']}"

        # Drop 'app' and 'psize' to get data for the bar chart
        data = row[numeric_cols]
        #print(data)

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        plt.subplots_adjust(left=0.15, bottom=0.3)
        data = data[['Migr.', 'Host', 'Dev+Migr.', 'Dev+Host', 'MIM']]
        data.plot(kind='bar', color='skyblue', edgecolor='black')

        # Formatting
        #plt.title(title)
        plt.xlabel("Strategies")
        plt.ylabel("Speedup (vs Migr)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Show the plot
        os.makedirs(f"../../figs/{hostname}/perf_comparison/{row['app']}", exist_ok=True)
        plt.savefig(f"../../figs/{hostname}/perf_comparison/{row['app']}/{row['app']}_{row['psize']}.png")
        plt.close()

    #Avg Barchart 
    #avg_row = df[['Migr.', 'Host', 'Dev+Migr.', 'Dev+Host', 'Static-Opt', 'MIM']].aggregate(gmean)
    avg_row = df[['Migr.', 'Host', 'Dev+Migr.', 'Dev+Host', 'MIM']].aggregate(gmean)
    colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
    hatch_patterns = ['/', '\\', 'x', '-', 'o']
    plt.figure(figsize=(10, 4))
    #plt.subplots_adjust(left=0.07, bottom=0.20)
    
    ax = avg_row.plot(kind='bar', color=colors, edgecolor='black', hatch=hatch_patterns)

    # Formatting
    #plt.title(title)
    #plt.xlabel("Strategies")
    plt.hlines(1,-0.5,5.5)
    plt.ylabel("Speedup (vs Migr.)")
    plt.yticks([1, 2, 3])
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Define hatching patterns

    # Apply hatching
    #for i, bar_container in enumerate(ax.containers):
    #    for bar in bar_container:
    #        bar.set_hatch(hatch_patterns[i % len(hatch_patterns)])

    # Annotate each bar with its value
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(), 2)), 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')

    # Show the plot
    plt.tight_layout()
    os.makedirs(f"../../figs/{hostname}/perf_comparison/", exist_ok=True)
    plt.savefig(f'../../figs/{hostname}/perf_comparison/default_{kernel_version}_vanilla_geomean_all.png')
    plt.close()
    

def make_accuracy_plot(csv_file):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, header=0)
    # Ensure required columns exist
    if 'app' not in df or 'psize' not in df:
        raise ValueError("CSV must contain 'app' and 'psize' columns.")

    df = df[df['app'].isin(benchmarks)]
    df = df[df['psize'].isin(psizes)]
    # Convert numeric columns to float
    numeric_cols = df.columns.difference(['app', 'psize'])
    df[numeric_cols] = df[numeric_cols].astype(float)
    df[numeric_cols] = df[numeric_cols].div(df['Opt'], axis=0) #gives slowdown from optimal
    df[numeric_cols] = df[numeric_cols].map(lambda x: 1 - x)
    
    # Thresholds
    thresholds = np.linspace(0, 1, 200)

    # Count how many values in each column are greater than each threshold
    df_counts = pd.DataFrame(
        {t: (df[numeric_cols] < t).sum() / solution.size for t in thresholds}
    )
    df_counts = df_counts.T
    
    # Set up the plot
    strategies = ["MIM", "Dev+Host", "Dev+Migr.", "Migr.", "Host"]
    colors = ["#0072B2", "#F0E442", "#009E73", "#E69F00", "#56B4E9"]
    linestyle = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))]
    
    # Plot
    textsize = 24
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(10, 6))
    for i, strat in enumerate(strategies):
        plt.plot(thresholds, df_counts[strat], label=strat, color=colors[i], linestyle=linestyle[i], linewidth=4)
    
    #Place an X where MIM reaches 1
    s = df_counts["MIM"]    
    arr = s.to_numpy()  # or s.values
    pos = np.searchsorted(arr, 1, side='left')
    plt.plot([pos / 200], [1], color=colors[4], marker='x', markersize=16)
    
    plt.xlabel('Minimum Slowdown from Optimal')
    plt.ylabel('Fraction of Policies')
    plt.grid(True)
    plt.legend(title="Strategy", fontsize=18, title_fontsize=18)#, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5, columnspacing=0.5, handletextpad=0.25)
    plt.tight_layout()
    os.makedirs(f"../../figs/{hostname}/perf_comparison/", exist_ok=True)
    plt.savefig(f'../../figs/{hostname}/perf_comparison/default_{kernel_version}_vanilla_slowdown_all.png')
    plt.close()

   
def make_radar_chart(ax, columns, values, title):
    num_vars = len(columns)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the circle
    
    values += values[:1]  # Close the circle
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columns, fontsize=base)  # Ensuring labels show 
   
    #Standard scale
    if(True):
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels([0.25, 0.5, 0.75], fontsize=base-12)  # Ensuring labels show 
        ax.set_ylim(0, 1)
    
    #Log scale 
    else:
        values = [np.log2(e) for e in values]
        ax.set_yticks(np.log2([0.015, 0.06, 0.25, 1]))
        ax.set_yticklabels([0.0015, 0.06, 0.25, 1], fontsize=base-8)  # Ensuring labels show 
        ax.set_ylim(np.log2(0.0015), 0)
    
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    ax.set_title(title, fontsize=base+2)

def radar_plot(df):
    plt.rcParams['text.usetex'] = True
    #df['dr_intra_mean_1000_norm'] = df['dr_intra_mean_1000'] / df['dr_intra_mean_1000'].max()
    #df['dr_intra_mean_1_norm'] = df['dr_intra_mean_1'] / df['dr_intra_mean_1'].max()


# Group by (app, psize)
    for (app, psize), group in df.groupby(["app", "psize"]):
        labels = group["label"].unique()
        num_labels = len(labels)
        
        fig, axes = plt.subplots(1, num_labels, subplot_kw={'projection': 'polar'}, figsize=(5 * num_labels, 6))
        if num_labels == 1:
            axes = [axes]  # Ensure iterable
        
        for ax, label in zip(axes, labels):
            columns = [
                "tr_median_1000", "ts_rel_median_1000", "d_mean_1000", 
                "dc_intra_rel_mean_1000", "dr_intra_mean_1000", "tr_mean_1"
            ]
            column_labels = [r'$\hat{S_{[1000]}}$', r'$\overline{S}$', r'$\hat D$', r'$\overline{\mbox{fdup}}$', r"$\hat{\mbox{fdup}}$", r'$\hat{S_{[1]}}$']
            subset = group[group["label"] == label]
            mean_values = [max(e, 10e-6) for e in subset[columns].mean().tolist()]
            make_radar_chart(ax, column_labels, mean_values, title=f'{label.split(".")[0].lower()}\\%')
        
        #plt.suptitle(f'App: {app}, Psize: {psize}', fontsize=base+4)
        plt.tight_layout(pad=1.6, rect=[0, -0.06, 1, 1.06])
        os.makedirs(f"../../figs/{hostname}/metrics_radar/", exist_ok=True)
        plt.savefig(f'../../figs/{hostname}/metrics_radar/default_{kernel_version}_faults-new_{psize}_{app}.png')
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subset", action="store_true", help="Only compute a representative subset of data")
    args = parser.parse_args()

    if args.subset:
        benchmarks = ['FDTD-2D', 'stream', 'sgemm', 'bfs-worst']
        psizes = [100, 125]
    
    #Parse data from external files 
    full_df = parse_df()
    pf = parse_perf_df()
    solution = get_solution(benchmarks, psizes, pf.copy())
    print_strat_naive(full_df.copy(), pf.copy(), p=False)
    make_accuracy_plot("paper_metrics.csv")
    make_perf_barchart("paper_metrics.csv")
    make_all_perf_barchart("paper_metrics.csv")
    radar_plot(full_df.copy())
            

    
