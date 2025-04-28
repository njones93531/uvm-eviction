import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
import heapq
from itertools import combinations
from scipy.stats import chi2_contingency

# Define the path to the main directory
main_directory = '../../figs/voltron/metrics_stats_relative/'
perf_data_base_dir = '/home/najones/uvm-eviction/benchmarks/strategied'
metric_stats_type = 'default'
DEVICE_SIZE = 11.6
benchmarks = ['FDTD-2D', 'GRAMSCHM', 'stream', 'cublas', 'bfs-worst']
pd.options.display.max_rows = 999
pd.set_option('display.max_columns', 999)
enable_good_OR = False
enable_OSOR = False
enable_OR = True
enable_RATIO = False
ENABLE_DR_COMBOS = False

solution = pd.DataFrame({
        'app': ['bfs-worst', 'cublas', 'FDTD-2D', 'GRAMSCHM', 'stream', 'tealeaf'],
        #9.0  : ['ddd', 'ddd', 'ddd', 'ddd', 'ddd'],
        12.0 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd', 'x'*20],
        15.0 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd', 'x'*20],
        18.0 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm', 'x'*20],
        21.0 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm', 'x'*20]}).set_index('app')
    

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
    print(f"Using metric {metric}")
    print(pivot_table)

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
    psizes = [12.0, 15.0, 18.0, 21.0]

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(0.0, index=apps, columns=psizes)
    comparison_result.index.name = 'app'
    comparison_default = pd.DataFrame(0.0, index=apps, columns=psizes)
    comparison_default.index.name = 'app'
    score = [0, 0]
    
    #Problem Size,Policy,Iteration,Kernel Time
    # Iterate through each app and psize and compare the strategy strings
    for app in apps:
        for psize in psizes:
            strategy1 = ""
            cell_view = None
            no_value = False
            try:
                strategy1 = str(pivot1.loc[app, psize])
                cell_view = perf_df[(perf_df['app'] == app) & (perf_df['Problem Size'] == int(psize))].groupby(['app', 'Problem Size', 'Policy'], observed=True).mean().reset_index()
                cell_view_default_time = float(cell_view[cell_view['Policy'] == 'm' * len(strategy1)]['Kernel Time'].iloc[0])
                cell_view_best_time = float(cell_view[cell_view['Kernel Time'] > 0]['Kernel Time'].min())
                cell_view_best_policy = str(cell_view[cell_view['Kernel Time'] == cell_view_best_time].iloc[0])
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
                comparison_result.loc[app, psize] = None
                comparison_default.loc[app, psize] = None
            elif not check_strat(strategy1, cell_view_best_policy):
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
    psizes = [12.0, 15.0, 18.0, 21.0]

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(False, index=apps, columns=psizes)
    comparison_result.index.name = 'app'
    score = 0
    pp = psizes#[12.0, 15.0]
    aa = apps#['stream', 'cublas']#'bfs-worst', 'FDTD-2D', 'GRAMSCHM']#'cublas']
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
                
                if app not in ['tealeaf', 'spmv-coo-twitter7', 'GEMM'] and 'nopf' not in file and psize != 9.0: #Ignore spmv for now (const psize)
                # Add a new columns
                    data_df['app'] = app
                    data_df['psize'] = psize
                    data_df['sub_ratio'] = psize / DEVICE_SIZE * 100
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

def classify_d_with(data_df, classifier, scalar, p=False, apply_m=False, perf_df=pd.DataFrame()):
    # Initialize an empty list to store DataFrames

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
    return -1 * score2[0]




def classify_d_with_legacy(data_df, classifier, scalar, mode, p, invert=False, apply_m=False, perf_df=pd.DataFrame()):
    # Initialize an empty list to store DataFrames

    if isinstance(data_df[classifier].iloc[0], str):
        print(f"Invalid classifier: {classifier}")
        return -1000
    if scalar == 1:
        data_df["new"] = data_df[classifier]
    if scalar == 0: 
        data_df["new"] = (data_df[classifier] * -1) + data_df[classifier].max()
    if scalar == -1:
        data_df["new"] = data_df[classifier] * -1
    
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
        #If threshold is high, use t
        #d first
        capacity = .99
        chosen = []
        #if mode == "knapsack":
        chosen = knapsack(tmp, capacity, 'size', 'new')
        #else:
         #   chosen = migrate_me(tmp, capacity, 'size', 'new')
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
    #print(f"p is {p}")
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
            plt.savefig('/home/najones/uvm-eviction/figs/voltron/correlation_matrix_x.png')


    #Refine parameters
    corr_matrix_d = []
    if data_df['d'].nunique != 1:
        corr_matrix_d = filter_corr_matrix_by_threshold(data_df.copy(), 0.10, 'd')
        #corr_matrix_d = corr_matrix_d.to_pandas()
        if(p):
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix_d, annot=True, cmap='coolwarm', cbar=True)
            plt.savefig('/home/najones/uvm-eviction/figs/voltron/correlation_matrix_d.png')

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
            plt.savefig('/home/najones/uvm-eviction/figs/voltron/correlation_matrix_hm.png')
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
    plt.savefig('/home/najones/uvm-eviction/figs/voltron/small_correlation_matrix_metrics_d.png')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix_hm, annot=True, cmap='coolwarm', cbar=True)
    plt.savefig('/home/najones/uvm-eviction/figs/voltron/small_correlation_matrix_metrics_hm.png')

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
        print("Using oracle m prediction")
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

def classify_both(df_data, df_perf, d_predictor, m_predictors, p=False):
    set_m_prediction(df_data, m_predictors, p)
    return classify_d_with(df_data, d_predictor[0], d_predictor[1], p=p, apply_m=True, perf_df=df_perf)

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
    #get_smallest_corr_matrix(df.copy())
    #best_d_metrics = select_columns_with_strong_relationship(corr_d, 'd', max_cols=9, threshold=1.1)
    #best_m_metrics = select_columns_with_strong_relationship(corr_m, 'm', max_cols=5, threshold=0.9)
    #best_x_metrics = select_columns_with_strong_relationship(corr_x, 'x', max_cols=100, threshold=0.99)
    #print("Besst x metrics: ", best_x_metrics)
    #try_fdtd_vs_stream(df, best_x_metrics)
    #m_dict, bfm_cols = rank_m_h_classifiers(df.copy(), p)
    #d_dict, bfm_cols = rank_d_notd_classifiers(df.copy())
    #m_tuples = [m_dict[x] for x in best_m_metrics if m_dict[x][1] > 0]
    #best_m_tuples = m_tuples#[m_dict[x] for x in best_m_metrics if m_dict[x][1] > 0]
    #other_m_tuples = bfm_cols[:9]
    #score_row_m_classifier(df.copy())
    #rank_d_notd_classifiers(df.copy())
    #exit(0)
    #d_dict, brute_force_best, best_ds = brute_force_metrics_vs_d(df.copy(), metrics=best_d_metrics, perf_df=pf.copy(), p=p)
    d_dict, brute_force_best, best_ds = brute_force_metrics_vs_d(df.copy(), metrics=[], perf_df=pf.copy(), p=p)
    #d_tuples = [d_dict[x] for x in best_d_metrics]
    #best_d_tuple = sorted(d_tuples, key=lambda x: x[1])[-1][0]
    best_d_tuple = brute_force_best
    #best_d_tuple = ('tr_mean_1000_OR_tr_prop_0s_1', 1, 'knapsack', False)
    #print("Optimizing ensemble")
    #best_m_tuples = brute_force_find_ensemble(df.copy(), best_d_tuple, m_tuples, pf.copy())
    #best_m_tuples = m_tuples
    #best_d_tuple = ('d_prop_max_1000_max_tr_mean_1000', 1, 'knapsack', False)
    #print("Optimizing ensemble again")
    #best_bf_m_tuples = brute_force_find_ensemble(df.copy(), best_d_tuple, other_m_tuples, pf.copy())
    #best_bf_m_tuples = other_m_tuples
    #oracle_d_tuples = [('d', 1, 'migrate', False, False) for i in range(0, 8)]
    #oracle_d_tuples = [('d', 0.5, True) for i in range(0, 8)]
    #oracle_m_tuples = [('m', 0.5, True) for i in range(0, 8)]
    #print("-"*80)
    #print("M only")
    #vote_for_both(df.copy(), pf.copy(), oracle_d_tuples, m_tuples)
    #print("-"*80)
    #print("D only")
    #vote_for_both(df.copy(), pf.copy(), d_tuples, oracle_m_tuples)
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
    dr_condition = app_df['dr_intra_mean_1000'].between(1, 256, inclusive="both").all()
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

def predict_strat_hardcode(data_df, pf):
    classify_apps_hardcode(data_df)
    
    df1 = data_df[data_df['x_class'] == True]
    print_metric_abc(df1.copy(), 'd_mean_1000')
    m_metric = ('d_mean_1000', 0.13, True)
    d_metric = ('dc_intra_rel_mean_1', -1)
    classify_both(df1.copy(), pf.copy(), d_metric, [m_metric], p=True)
 
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


if __name__ == "__main__":
    #Parse data from external files 
    full_df = parse_df()
    pf = parse_perf_df() 

    #classify_apps_hardcode(full_df)
    #predict_strat_hardcode(full_df.copy(), pf.copy())
    #exit()

    #data_df = full_df[full_df['app'].isin(['tealeaf', 'conjugateGradientUM', 'hpgmg'])]
    #predict_unseen_strat(data_df)
    #exit()
    #Separate allocation data from application-wide data
    #app_df = full_df[full_df['label'] == 'all']
    #full_df = full_df[full_df['label'] != 'all']
   
    """
    for app in benchmarks: 
        df1 = full_df[full_df['app'] == app]
        print("Ranking d classifiers")
        d_dict, brute_force_best, best_mets = brute_force_metrics_vs_d(df1.copy(), metrics=[], perf_df=pf.copy())
        classify_both(df1.copy(), pf.copy(), brute_force_best, ['m'], p=True)
        #classify_both(df1.copy(), pf.copy(), brute_force_best, ['m'], p=True)
        #print('mh mets')
        #print(bfm_cols[:3])
        print('d met')
        printall(best_mets) 
        print('-'*40)

    exit()
    """
    #Differentiate between streaming patterns and others 
    #print_apps_hardcode(full_df.copy())
    #classify_apps_hardcode(full_df)

    #Separate allocation data from application-wide data
    #app_df = full_df[full_df['label'] == 'all']
    #full_df = full_df[full_df['label'] != 'all']
    #df1 = full_df
    
    df1 = full_df[full_df['x'] == True]

    #print("Ranking m/h classifiers")
    #m_dict, bfm_cols = rank_m_h_classifiers(df1.copy(), p=False)
    print("Ranking d classifiers")
    d_dict, brute_force_best, best_mets = brute_force_metrics_vs_d(df1.copy(), metrics=[], perf_df=pf.copy(), p=True)
    #classify_both(df1.copy(), pf.copy(), brute_force_best, bfm_cols[:1], p=True)
    classify_both(df1.copy(), pf.copy(), brute_force_best, ['m'], p=True)
    exit()
    #print('mh mets')
    #printall(bfm_cols)
    print('d met')
    #print(brute_force_best)
    printall(best_mets) 
    print('-'*40)
   
    
    df2 = full_df[full_df['x'] == False]

    #print("Ranking m/h classifiers")
    #m_dict, bfm_cols = rank_m_h_classifiers(df2.copy(), p=True)
    print("Ranking d classifiers")
    d_dict, brute_force_best, best_mets = brute_force_metrics_vs_d(df2.copy(), metrics=[], perf_df=pf.copy(), p=False)
    classify_both(df2.copy(), pf.copy(), brute_force_best, ['m'], p=True)
    #classify_both(df2.copy(), pf.copy(), brute_force_best, ['m'], p=True)
    #print('mh mets')
    #printall(bfm_cols)
    print('d met')
    #print(brute_force_best)
    printall(best_mets) 
    print('-'*40)

    exit()
    df2 = full_df[full_df['x'] == False]
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df2.copy(), p=True)
    #d_dict, brute_force_best, best_mets = brute_force_metrics_vs_d(df2.copy(), metrics=[], perf_df=pf.copy())
    #classify_both(df2.copy(), pf.copy(), brute_force_best, bfm_cols[:3], p=True)
    #classify_both(df2.copy(), pf.copy(), brute_force_best, ['m'], p=True)
    print('mh mets')
    printall(bfm_cols)
    #print('d met')
    #printall(best_mets)

    print('-'*40)
    exit()




    full_df = parse_df()
    pf = parse_perf_df()

    print_apps_hardcode(full_df.copy())
    classify_apps_hardcode(full_df)
    app_df = full_df[full_df['label'] == 'all']
    full_df = full_df[full_df['label'] != 'all']

    #df1 = full_df[full_df['app'] == 'stream']

    df1 = full_df[full_df['x_class'] == True]
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df1.copy())
    d_dict, brute_force_best, _ = brute_force_metrics_vs_d(df1.copy(), metrics=[], perf_df=pf.copy())
    classify_both(df1.copy(), pf.copy(), brute_force_best, bfm_cols[:3], p=True)
    print('mh mets')
    print(bfm_cols[:3])
    print('d met')
    print(brute_force_best)

    df2 = full_df[full_df['x_class'] == False]
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df2.copy())
    d_dict, brute_force_best, _ = brute_force_metrics_vs_d(df2.copy(), metrics=[], perf_df=pf.copy())
    classify_both(df2.copy(), pf.copy(), brute_force_best, bfm_cols[:3], p=True)
    print('mh mets')
    print(bfm_cols[:3])
    print('d met')
    print(brute_force_best)
    exit()


    full_df = parse_df()
    pf = parse_perf_df() 


    
   
    #print_apps_hardcode(full_df.copy())
    #exit()
    #classify_apps_hardcode(full_df)
    app_df = full_df[full_df['label'] == 'all']
    full_df = full_df[full_df['label'] != 'all']
    
    """
    df1 = full_df[full_df['x_class'] == True]
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df1.copy())
    d_dict, brute_force_best, _ = brute_force_metrics_vs_d(df1.copy(), metrics=[], perf_df=pf.copy())
    classify_both(df1.copy(), pf.copy(), brute_force_best, bfm_cols[:3], p=True)
    print('mh mets')
    print(bfm_cols[:3])
    print('d met')
    print(brute_force_best)
    
    df2 = full_df[full_df['x_class'] == False]
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df2.copy())
    d_dict, brute_force_best, _ = brute_force_metrics_vs_d(df2.copy(), metrics=[], perf_df=pf.copy())
    classify_both(df2.copy(), pf.copy(), brute_force_best, bfm_cols[:3], p=True)
    print('mh mets')
    print(bfm_cols[:3])
    print('d met')
    print(brute_force_best)
    exit()
    """
    #classify_m_given_d(df_data, df_perf, m_predictors, p=False)
    #scores = []
    train_classify_app_group(app_df, ['FDTD-2D', 'stream'])
    
    ds = {}
    #Try each individual
    #for app in benchmarks:
        #print(app)
        #df1 = full_df[full_df['app'] == app]
        #ds[app] = score_df(df1, pf.copy(), p=False)
    #scores = sorted(scores, key=lambda x: x[1])
    exit()
    similarity = {}
    for combo in [('FDTD-2D', 'stream'), ('GRAMSCHM', 'cublas', 'bfs-worst')]:
        tmp = set(ds[combo[0]])
        for i in range(1, len(combo)):
            tmp = tmp & set(ds[combo[i]])
        similarity[combo] = len(tmp)
        print(combo)
        print(tmp)
    
    exit()


    scores = []
    ds = {}
    #Try each individual
    for app in benchmarks:
        print(app)
        df1 = full_df[full_df['app'] == app]
        ds[app] = score_df(df1, pf.copy(), p=False)
    #scores = sorted(scores, key=lambda x: x[1])
    similarity = {}
    for j in range(1, len(benchmarks)+1):
        for combo in combinations(benchmarks, j):
            tmp = set(ds[combo[0]])
            for i in range(1, len(combo)):
                tmp = tmp & set(ds[combo[i]])
            similarity[combo] = len(tmp)
            if j == 5:
                print("The winners:")
                for s in tmp:
                    print(s)
    print(similarity)
    
    exit()

    for i in range(2):
        for combo in combinations(benchmarks, i+1):
            print(f"Black sheeps: {combo}")
            applist = combo
            df1 = full_df[~full_df['app'].isin(applist)]
            df2 = full_df[full_df['app'].isin(applist)]
            scores.append((combo, score_mutex_dfs(df1, df2, pf.copy())))
    scores = sorted(scores, key=lambda x: x[1])
    print(scores)

    applist = scores[-1][0]
    df1 = full_df[~full_df['app'].isin(applist)]
    df2 = full_df[full_df['app'].isin(applist)]
    score_mutex_dfs(df1, df2, pf.copy(), p=True)
    exit()
    no_streamfdtd = full_df[~full_df['app'].isin(['stream', 'FDTD-2D'])]
    streamfdtd = full_df[full_df['app'].isin(['stream', 'FDTD-2D'])]
    for df in [streamfdtd, no_streamfdtd]:
        print("Calculating Correlation Matrices")
        corr_d, corr_m, corr_x = get_corr_matrix(df.copy())
        #get_smallest_corr_matrix(df.copy())
        best_d_metrics = select_columns_with_strong_relationship(corr_d, 'd', max_cols=5, threshold=0.9)
        best_m_metrics = select_columns_with_strong_relationship(corr_m, 'm', max_cols=5, threshold=0.9)
        #best_x_metrics = select_columns_with_strong_relationship(corr_x, 'x', max_cols=100, threshold=0.99)
        #print("Besst x metrics: ", best_x_metrics)
        #try_fdtd_vs_stream(df, best_x_metrics)
        print("Ranking m/h classifiers")
        m_dict, bfm_cols = rank_m_h_classifiers(df.copy())
        #d_dict, bfm_cols = rank_d_notd_classifiers(df.copy())
        m_tuples = [m_dict[x] for x in best_m_metrics if m_dict[x][1] > 0]
        best_m_tuples = m_tuples#[m_dict[x] for x in best_m_metrics if m_dict[x][1] > 0]
        other_m_tuples = bfm_cols[:9]
        #score_row_m_classifier(df.copy())
        #rank_d_notd_classifiers(df.copy())
        print("Ranking d classifiers")
        #exit(0)
        d_dict, brute_force_best, _ = brute_force_metrics_vs_d(df.copy(), metrics=best_d_metrics, perf_df=pf.copy())
        #d_dict, brute_force_best = brute_force_metrics_vs_d(df.copy(), metrics=[], perf_df=pf.copy())
        d_tuples = [d_dict[x] for x in best_d_metrics]
        best_d_tuple = sorted(d_tuples, key=lambda x: x[1])[-1][0]
        best_d_tuple = brute_force_best
        #print("Optimizing ensemble")
        #best_m_tuples = brute_force_find_ensemble(df.copy(), best_d_tuple, m_tuples, pf.copy())
        #best_m_tuples = m_tuples
        #best_d_tuple = ('d_prop_max_1000_max_tr_mean_1000', 1, 'knapsack', False)
        #print("Optimizing ensemble again")
        #best_bf_m_tuples = brute_force_find_ensemble(df.copy(), best_d_tuple, other_m_tuples, pf.copy())
        #best_bf_m_tuples = other_m_tuples
        #oracle_d_tuples = [('d', 1, 'migrate', False, False) for i in range(0, 8)]
        #oracle_d_tuples = [('d', 0.5, True) for i in range(0, 8)]
        #oracle_m_tuples = [('m', 0.5, True) for i in range(0, 8)]
        #print("-"*80)
        #print("M only")
        #vote_for_both(df.copy(), pf.copy(), oracle_d_tuples, m_tuples)
        #print("-"*80)
        #print("D only")
        #vote_for_both(df.copy(), pf.copy(), d_tuples, oracle_m_tuples)
        print("-"*30)
        print("Our classification")
        classify_both(df.copy(), pf.copy(), best_d_tuple, best_m_tuples)
        print("m tuples:\n", best_m_tuples)
        print("d tuples:\n", best_d_tuple)
        #write_metrics(df.copy(), best_d_tuple, best_m_tuples)
    
    #print("-"*80)
    #print("BF classification")
    #classify_both(df.copy(), pf.copy(), best_d_tuple, best_bf_m_tuples)
    #print("best m tuples:\n", best_bf_m_tuples)
    #print("d tuples:\n", d_tuples)
    #print("BF Best :\n", best_d_tuple)
    #score = classify_d_with(df.copy(), 'tr_prop_1000', 0, 'knapsack', True, False)
        

    
