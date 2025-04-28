import os
import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# Define the path to the main directory
main_directory = '../../figs/voltron/metrics_stats/'
perf_data_base_dir = '../../benchmarks/strategied'
DEVICE_SIZE = 12.
benchmarks = ['FDTD-2D', 'GRAMSCHM', 'stream', 'cublas', 'bfs-worst']
pd.options.display.max_rows = 999
pd.set_option('display.max_columns', 999)


solution = pd.DataFrame({
        'app': ['bfs-worst', 'cublas', 'FDTD-2D', 'GRAMSCHM', 'stream'],
        #9.0  : ['ddd', 'ddd', 'ddd', 'ddd', 'ddd'],
        12.0 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd'],
        15.0 : ['dhd', 'mdd', 'hdd', 'dmd', 'mdd'],
        18.0 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm'],
        21.0 : ['hdd', 'mmd', 'hhd', 'dmh', 'dmm']}).set_index('app')
    

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
    corr_matrix = df.corr()
    return corr_matrix
    

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

        




def knapsack(df, cap, cost_str, val_str, invert=False):
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

    # Loop over all possible combinations of items
    for r in range(1, n + 1):
        for combo in combinations(range(n), r):
            total_cost = sum(costs[i] for i in combo)
            #In some cases, we care more about what we leave off the GPU
            total_value = 0
            if invert: 
                total_value = sum(values) - sum(values[i] for i in combo)
            else:
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
    apps = pivot1.index.astype(str)    
    psizes = pivot1.columns.astype(float)

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(0.0, index=apps, columns=psizes)
    comparison_default = pd.DataFrame(0.0, index=apps, columns=psizes)
    score = [0, 0]
    
    #Problem Size,Policy,Iteration,Kernel Time
    # Iterate through each app and psize and compare the strategy strings
    for app in apps:
        for psize in psizes:
            strategy1 = str(pivot1.loc[app, psize])
            cell_view = perf_df[(perf_df['app'] == app) & (perf_df['Problem Size'] == int(psize))].groupby(['app', 'Problem Size', 'Policy'], observed=True).mean().reset_index()
            no_value = False
            try:
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
    apps = pivot1.index.astype(str)    
    psizes = pivot1.columns.astype(float)

    # Create an empty DataFrame to store the comparison results
    comparison_result = pd.DataFrame(False, index=apps, columns=psizes)
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
            data_df = pd.read_csv(file_path, dtype=dtype_dict)
            app = str(file).split('_')[-1].split('.')[0]
            psize = float(str(file).split('_')[-2])
            
            if app not in ['spmv-coo-twitter7', 'GEMM'] and 'nopf' not in file and psize != 9.0: #Ignore spmv for now (const psize)
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
                #data_df['space_taken_by_placement'] = data_df['d'].astype(float) * data_df['size'] / 100.0
                #data_df['space_taken_by_placement'] = data_df['space_taken_by_placement'].sum()
                #data_df['space_remaining'] = 1 - data_df['space_taken_by_placement']
                data_df['size'] = data_df['size'] / 100.0
                data_df['anti_size'] = 1 - data_df['size']
                #New cols of normal floats
                #for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                #    data_df[f'{f}'] = f

                #Add new metric, the max of two other metrics 
                cols = data_df.columns
                cols = set(cols) - set(['strat', 'd', 'h', 'm', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
                cols = sorted([c for c in cols if numeric(data_df[c][0])])
                
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
                cols = sorted([c for c in cols if numeric(data_df[c][0])])
                tmp_dfs = []
                good_metrics = ['tr_mean_1000', 'tr_median_1', 'd_max_1000_OR_ws_prop_max_1000', 'd_mean_1000_OR_tr_mean_1000', 'd_median_1000_OR_tr_mean_1000', 'd_median_1000_OR_tr_median_1000', 'd_prop_0s_1_OR_tr_mean_1000', 'd_prop_0s_1_OR_tr_median_1000', 'd_prop_0s_1000_OR_tr_mean_1000', 'd_prop_0s_1000_OR_tr_median_1000', 'd_prop_1_OR_tr_median_1000', 'tr_mean_1_OR_tr_mean_1000', 'tr_mean_1_OR_tr_prop_1000', 'tr_mean_1_OR_ws_prop_max_1000', 'tr_mean_1000_OR_tr_median_1', 'tr_mean_1000_OR_tr_median_1000', 'tr_mean_1000_OR_tr_prop_0s_1', 'tr_mean_1000_OR_tr_prop_0s_1000', 'tr_mean_1000_OR_tr_prop_1', 'tr_mean_1000_OR_tr_prop_1000', 'tr_mean_1000_OR_tr_prop_max_1000', 'tr_mean_1000_OR_ws_prop_0s_1', 'tr_mean_1000_OR_ws_prop_0s_1000', 'tr_mean_1000_OR_ws_prop_max_1000', 'tr_median_1_OR_tr_prop_1000', 'tr_median_1_OR_ws_prop_max_1000', 'tr_median_1000_OR_tr_prop_0s_1', 'tr_median_1000_OR_tr_prop_0s_1000', 'tr_median_1000_OR_tr_prop_1', 'tr_median_1000_OR_tr_prop_1000', 'tr_median_1000_OR_tr_prop_max_1000', 'tr_median_1000_OR_ws_prop_0s_1', 'tr_median_1000_OR_ws_prop_max_1000', 'tr_prop_max_1000_OR_ws_prop_1']
                stream_fdtd_metrics = ['d_prop_0s_1_OR_tr_mean_1000', 'd_prop_0s_1_OR_tr_mean_1000', 'd_prop_0s_1_OR_tr_mean_1', 'tr_mean_1_OR_tr_prop_0s_1000', 'dr_median_1_OR_sub_ratio', 'tr_mean_1_OR_tr_prop_1000', 'tr_mean_1000_OR_ws_prop_max_1000']
                #for metric2 in good_metrics:#cols:
                #    tmp_df = pd.DataFrame(index=data_df.index)
                #    for metric1 in stream_fdtd_metrics:
                #        tmp_df[f'{metric2}_minus_{metric1}'] = data_df[metric2] - data_df[metric1]
                #        #tmp_df[f'{metric2}_plus_{metric1}'] = data_df[metric2] + data_df[metric1]
                #        #tmp_df[f'{metric2}_ORR_{metric1}'] = data_df[[metric1, metric2]].max(axis=1)
                #    tmp_dfs.append(tmp_df)
                #data_df = pd.concat([data_df] + tmp_dfs, axis=1)

                dfs.append(data_df)
                print("File complete")

    # Concatenate all the DataFrames together
    combined_df = pd.concat(dfs, ignore_index=True)
    print("Finished parsing metric files")
    return combined_df

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
    except ValueError:
        return False

def rank_d_notd_classifiers(data_df):
    scores_to_dividers = {}
    metrics_to_tuples = {}
    for metric in data_df.columns:
        if numeric(data_df[metric][0]):
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



def rank_m_h_classifiers(data_df):
    data_df = data_df[data_df['d'] == False]
    cols = data_df.columns
    cols = [c for c in cols if 'OR' not in c]
    scores_to_dividers = {}
    metrics_to_tuples = {}
    for metric in data_df.columns:
        if numeric(data_df[metric][0]) and metric in cols:
            best_score, best_divider, m_is_up = find_border_h_m(data_df, metric)
            metrics_to_tuples[metric] = (metric, best_divider, m_is_up)
            if best_score in scores_to_dividers:
                scores_to_dividers[best_score].append((metric, best_divider, m_is_up))
            else:
                scores_to_dividers[best_score] = [(metric, best_divider, m_is_up)]
    max_score = max(list(scores_to_dividers.keys()))
    print(f"Max score is {max_score}\nThese metrics achieve it:")
    print(scores_to_dividers[max_score])
    print("These were close:")
    print("Off by 1:\n", scores_to_dividers[max_score - 1])
    print("Off by 2:\n", scores_to_dividers[max_score - 2])
    columns = []
    classifiers = []
    data_df['vote'] = 0
    for leeway in [0, 1, 2]:
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
    print(data_df[columns + ['vote','m']])
    return metrics_to_tuples, classifiers


def predict_m(data_df):
    metric = 'ws_prop_max_1000_vs_space'
    data_df['predict_m'] = data_df[metric] < data_df['space_remaining']
    data_df['predict_m_score'] = data_df['predict_m'] & data_df['m']
    data_df = data_df.sort_values(metric)
    print(data_df[[metric, 'space_remaining', 'predict_m', 'predict_m_score']])

def score_row_m_classifier(data_df):
    data_df = data_df[data_df['d']==False]
    predict_m(data_df)
    data_df['predict_m_score'] = data_df['predict_m'] & data_df['m']
    score = data_df['predict_m_score'].astype(int).sum()
    print(score, len(data_df))


def classify_d_with(data_df, classifier, scalar, mode, p, invert=False, apply_m=False, perf_df=pd.DataFrame()):
    # Initialize an empty list to store DataFrames

    if isinstance(data_df[classifier][0], str):
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
        chosen = knapsack(tmp, capacity, 'size', 'new', False)
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

        data_df.loc[(data_df['app'] == idx[0]) & (data_df['psize'] == idx[1]), 'strategy'] = str(''.join(strategy))

    no_dups = data_df.drop_duplicates(subset=['app', 'psize'])
    # Pivot the DataFrame
    pivot_table = no_dups.pivot(index='app', columns='psize', values='strategy').sort_values(by='app', key=lambda x: x.str.lower())
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
    for col in df.columns.drop(target):
        correlation = df[col].corr(df[target])
        if abs(correlation) > threshold:
            selected_columns.append(col)

    # Add 'm' to the list of selected columns to keep it in the result
    selected_columns.append(target)

    # Filter the DataFrame to include only selected columns
    df_selected = df[selected_columns]
    return df_corr(df_selected)



def get_corr_matrix(data_df, p=False):
    #Drop non-numeric columns
    for column in data_df.columns:
        if not numeric(data_df[column][0]):
            data_df = data_df.drop(columns=[column])

    #Drop irrelevant columns
    data_df = data_df.drop(columns=['level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
    #data_df = mpd.DataFrame(data_df)

    #Refine parameters
    corr_matrix_d = filter_corr_matrix_by_threshold(data_df.copy(), 0.40, 'd')
    #corr_matrix_d = corr_matrix_d.to_pandas()
    if(p):
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix_d, annot=True, cmap='coolwarm', cbar=True)
        plt.savefig('/home/najones/uvm-eviction/figs/voltron/correlation_matrix_d.png')

    #Now do the h vs m columns, after d is eliminated
    data_df = data_df[data_df['d'] == False]
    cols = [c for c in data_df.columns if 'OR' not in c]
    data_df = data_df[cols]
    corr_matrix_hm = filter_corr_matrix_by_threshold(data_df.copy(), 0.30, 'h')
    #corr_matrix_hm = corr_matrix_hm.to_pandas()
    if(p):
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix_hm, annot=True, cmap='coolwarm', cbar=True)
        plt.savefig('/home/najones/uvm-eviction/figs/voltron/correlation_matrix_hm.png')
    return corr_matrix_d, corr_matrix_hm

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
    # Step 1: Get absolute correlations with the target column
    target_corr = corr_matrix[column_name].abs().sort_values(ascending=False)
    
    # Step 2: Drop the target column itself from the list
    target_corr = target_corr.drop(index=column_name)
    
    # Step 3: Select columns iteratively, ensuring no high correlation among them
    selected_columns = []
    blacklist = ['d', 'h', 'm']
    
    for col in target_corr.index:
        if len(selected_columns) >= max_cols:
            break
        
        # Check if the new column has a high correlation with any already selected column
        is_highly_correlated = any(corr_matrix[col].abs()[selected_col] > threshold for selected_col in selected_columns)
        
        if not is_highly_correlated and col not in blacklist:
            selected_columns.append(col)
    
    return selected_columns


def get_smallest_corr_matrix(data_df):
    
    corr_d, corr_m = get_corr_matrix(data_df.copy())
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

def brute_force_metrics_vs_d(data_df, metrics=[], perf_df=pd.DataFrame()):
    df = data_df.drop(columns=['strat', 'd', 'h', 'm', 'level_0','index','base_address','adjusted_base','adjusted_end','alloc'])
    if len(metrics) > 0:
        df = df[['app', 'psize'] + metrics]
    class_dict = {}
    metrics_to_tuples = {}
    classifiers = df.columns
    max_score = -10000000
    best = ""
    for i, classifier in enumerate(classifiers):
        if numeric(df[classifier][0]): 
            for m in [-1, 1]:
                for invert in [False]: #[True, False]
                    for mode in ["knapsack"]: #["knapsack", "migrate"]:
                        score = classify_d_with(data_df.copy(), classifier, m, mode, False, invert, False, perf_df)
                        metric_tuple = (classifier, m, mode, False, invert)
                        if classifier not in metrics_to_tuples:
                            metrics_to_tuples[classifier] = (metric_tuple, score)
                        elif score > metrics_to_tuples[classifier][1]:
                            metrics_to_tuples[classifier] = (metric_tuple, score)
                        class_dict[(classifier,m, mode, invert)] = score
                        if score > max_score:
                            max_score = score
                            best = (classifier, m, mode, invert)

    #print("classifier dict", class_dict)
    best_mets = []
    print('-' * 30)
    for key, val in class_dict.items():
        if val >= max_score -2:
            print(key, val)
        if val == max_score:
            best_mets.append(key)
            #best = key
            #classify_d_with(df.copy(), best[0], best[1], best[2], True, best[3])
    for metric in best_mets:
        print('-' * 30)
        print("best", metric)
        print(f"Classifying with {metric}")
        classify_d_with(data_df.copy(), metric[0], metric[1], metric[2], True, invert=metric[3], apply_m=False, perf_df=perf_df.copy())
    #score = classify_with_trio(df.copy())
    #print(score)
    return metrics_to_tuples, best_mets[0]

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
            


def set_m_prediction(data_df, m_predictors, p=True):
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

def classify_both(df_data, df_perf, d_predictor, m_predictors):
    set_m_prediction(df_data, m_predictors)
    classify_d_with(df_data, d_predictor[0], d_predictor[1], d_predictor[2], True, invert=d_predictor[3], apply_m=True, perf_df=df_perf)

def vote_for_both(df_data, df_perf, d_predictors, m_predictors):
    set_m_prediction(df_data, m_predictors)
    d_predictor = set_d_prediction(df_data, d_predictors)
    classify_d_with(df_data, d_predictor[0], d_predictor[1], d_predictor[2], True, invert=d_predictor[3], apply_m=True, perf_df=df_perf)


if __name__ == "__main__":
    df = parse_df()
    pf = parse_perf_df() 
    print("Calculating Correlation Matrices")
    corr_d, corr_m = get_corr_matrix(df.copy())
    get_smallest_corr_matrix(df.copy())
    best_d_metrics = select_columns_with_strong_relationship(corr_d, 'd', max_cols=3, threshold=0.9)
    best_m_metrics = select_columns_with_strong_relationship(corr_m, 'm', max_cols=9, threshold=0.9)
    print("Ranking m/h classifiers")
    m_dict, bfm_cols = rank_m_h_classifiers(df.copy())
    #d_dict, bfm_cols = rank_d_notd_classifiers(df.copy())
    m_tuples = [m_dict[x] for x in best_m_metrics if m_dict[x][1] > 0]
    other_m_tuples = bfm_cols[:9]
    #score_row_m_classifier(df.copy())
    #rank_d_notd_classifiers(df.copy())
    print("Ranking d classifiers")
    #exit(0)
    #d_dict, brute_force_best = brute_force_metrics_vs_d(df.copy(), metrics=best_d_metrics, perf_df=pf.copy())
    #d_dict, brute_force_best = brute_force_metrics_vs_d(df.copy(), metrics=[], perf_df=pf.copy())
    #d_tuples = [d_dict[x] for x in best_d_metrics]
    #best_d_tuple = sorted(d_tuples, key=lambda x: x[1])[-1][0]
    best_d_tuple = ('tr_mean_1000_OR_tr_prop_0s_1', 1, 'knapsack', False)
    print("Optimizing ensemble")
    #best_m_tuples = brute_force_find_ensemble(df.copy(), best_d_tuple, m_tuples, pf.copy())
    best_m_tuples = m_tuples
    #best_d_tuple = ('d_prop_max_1000_max_tr_mean_1000', 1, 'knapsack', False)
    print("Optimizing ensemble again")
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
    
    #print("-"*80)
    #print("BF classification")
    #classify_both(df.copy(), pf.copy(), best_d_tuple, best_bf_m_tuples)
    #print("best m tuples:\n", best_bf_m_tuples)
    #print("d tuples:\n", d_tuples)
    #print("BF Best :\n", best_d_tuple)
    #score = classify_d_with(df.copy(), 'tr_prop_1000', 0, 'knapsack', True, False)
        

    
