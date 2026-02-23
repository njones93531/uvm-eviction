import pandas as pd
from pathlib import Path
import argparse

def parse_metrics_relative_file(filepath):
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")

    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    return df

def app_is_streaming(app_df):
    dr_condition = app_df['dr_intra_mean_1000'].between(1, 1000, inclusive="both").all()
    tr_condition = app_df['tr_mean_1'].between(0, 0.005, inclusive="both").all()
    return dr_condition and tr_condition

def classify_apps_hardcode(data_df):
    x_class = False
    if app_is_streaming(data_df):
        x_class = True
    data_df['x_class'] = x_class

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
    data_df_view = data_df
    if(p):
        print("Hello")
        print(data_df_view[columns + ['m_prediction']])
        data_df.to_csv('metrics.csv', index=False)

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
    
    tmp = data_df.sort_values('label').reset_index(drop=True)
    strategy = ['-'] * len(tmp)
        
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

    data_df['strategy'] = strategy

    if(p):
        print(data_df[['label', 'strategy']])


def predict_unseen_strat(data_df):
    classify_apps_hardcode(data_df)
    print("This is x_class")
    print(data_df['x_class'])
    
    df1 = data_df[data_df['x_class'] == True]
    if not df1.empty:
        m_metric = ('d_mean_1000', 0.15, True)
        d_metric = ('dc_intra_rel_mean_1', -1)
        set_m_prediction(df1, [m_metric])
        print("This is m_prediction")
        print(df1['m_prediction'])

        #TODO Need to change this later, 
        #   right now this skips the knap
        #   we need to change 'none' to a column
        #   to optimize for
        d_predictor = ['none', 1]
        classify_d_with(df1, d_predictor[0], d_predictor[1], p=True, apply_m=True)
        return ''.join(df1['strategy'].values)

    else:
        df2 = data_df[data_df['x_class'] == False].copy()
        bc = 1000

        #Set m_prediction 
        df2['thold1'] = 0.02
        df2['thold2']= 1
        df2['cond1'] = df2[f'd_mean_{bc}'] > df2['thold1']
        df2['cond2'] = df2[f'dr_intra_mean_{bc}'] > df2['thold2']
        df2['m_prediction'] = df2['cond1'] & df2['cond2']
        d_metric = (f'tr_median_{bc}_OR_ts_rel_median_{bc}', 1)
        print("\nTesting d_metric")
        print(df2.head())
        print(df2['tr_median_{bc}_OR_ts_rel_median_{bc}'])
        classify_d_with(df2.copy(), d_metric[0], d_metric[1], p=True, apply_m=True)
        return ''.join(df2['strategy'].values)

    return None 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse metrics_stats_relative file into a DataFrame"
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to metrics_stats_relative .txt file",
    )

    args = parser.parse_args()

    df = parse_metrics_relative_file(args.filepath)

    print("\nParsed DataFrame:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    print("Final Strategy: " + predict_unseen_strat(df))
