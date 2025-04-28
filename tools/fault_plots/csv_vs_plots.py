import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def gen_fig(df, x_column, y_column, args):
    #Check that provided columns exist
    available_columns = df.columns.tolist()[1:]
    for column in [x_column, y_column]:
        if column not in available_columns:
            print("Specified column does not exist. Options for the given input file are:")
            print(available_columns)
            exit(1)


    # Define the output directory
    os.makedirs(f'{args.outdir}/{y_column}', exist_ok=True)
    outfile=os.path.join(f'{args.outdir}/{y_column}', f"{args.benchmark}_{y_column}_vs_{x_column}.png")


    # Group data by 'App_Name' and plot y_column vs x_column for each group
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    linestyles = ['-', ':', '--', '-.']
    colors = ["red", "blue", "green", "orange", "purple", "teal", "gold"]
    fig, ax = plt.subplots(figsize=(10,6))
    for i, (app_name, group) in enumerate(df.groupby('App_Name')):
        for j, (alloc_name, group2) in enumerate(group.groupby('Alloc_Name')):
            if args.allocations and 'ALL' not in alloc_name:
                ax.plot(group2[x_column], group2[y_column], marker=markers[j], color=colors[i], linestyle=linestyles[j%4], label=alloc_name)
            if not args.allocations and 'ALL' in alloc_name:
                ax.plot(group2[x_column], group2[y_column], marker=markers[j], color=colors[i], label=alloc_name)

    ax.set_xlabel(x_column.replace('_',' '))
    ax.set_ylabel(y_column.replace('_',' '))
    ax.legend(title='Alloc Name', loc='upper left')
    if args.logarithmic:
        plt.yscale('log')  # Set y-axis scale to logarithmic
    if args.xticks:
        plt.xticks(df[x_column].unique())
    plt.savefig(outfile)
    plt.close('all')

def main(args):
    # Read input data from the specified file
    df = pd.read_csv(args.input_file)

    #filter out ignoreable allocs
    df = df[~df['Alloc_Name'].isin(["IGNORE"])]

    #Add some additional columns
    df['Alloc_Name'] = df['App_Name'].astype(str) + '_' + df['Alloc_Name'].astype(str)
    df['Degree_of_Subscription'] = df['Problem_Size_(GB)'].astype(int) * 100 // 12
    df['Evictions_per_VABlock_Migrated'] = df['Total_Evictions'] / df['VABlocks_Migrated']
    df['Faultless_Eviction_Ratio'] = df['Total_Faultless_Evictions'] / df['Total_Evictions']
    df['Faults_per_Batch'] = df['Total_Faults'] / df['Total_Batches']
    df['VABlocks_Migrated_per_Batch'] = df['VABlocks_Migrated'] / df['Total_Batches']
    df['Faults_per_VABlock_Migrated'] = df['Total_Faults'] / df['VABlocks_Migrated']

    if args.benchmark != 'all':
        df = df[df['App_Name'] == args.benchmark]
   
    x_column = args.x_column 
    y_columns = []
    if args.all_possible_y:
        y_columns = df.columns.tolist()[1:]
    else:
        y_columns.append(args.y_column)

    for y_column in y_columns:
        if(x_column != y_column):
            gen_fig(df, x_column, y_column, args)

if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Plot specified columns from a CSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('x_column', type=str, help='Column name for the x-axis')
    parser.add_argument('y_column', type=str, help='Column name for the y-axis')
    parser.add_argument('-l', '--logarithmic', action='store_true', help='Sets Y axis as logarithmic') 
    parser.add_argument('-o', '--outdir', help='Output directory base', default="../fault_plots/figures/metrics/vs")
    parser.add_argument('-x', '--xticks', action='store_true', help='Set x ticks to unique x values in dataset')
    parser.add_argument('-a', '--allocations', action='store_true', help='Display allocation totals instead of application totals')
    parser.add_argument('-y', '--all_possible_y', action='store_true', help='Generate a graph for each possible y axis using the given x')
    parser.add_argument('-b', '--benchmark', help='Benchmark to plot', default='all')
    args = parser.parse_args()


    main(args)


