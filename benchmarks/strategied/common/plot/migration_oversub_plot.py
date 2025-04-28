import pandas as pd
import argparse 
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from matplotlib.cm import ScalarMappable


def groups(aoi, default):
    default = list(default)
    h = default.copy()
    h[aoi] = 'h'
    #d = default.copy()
    #d[aoi] = 'd'
    return ["".join(x) for x in [h]]

def get_aois(default):
    ret = []
    for i, c in enumerate(default):
        if c == 'm':
            ret.append(i)
    return ret

tick_fsize = 14
axis_fsize = 16
plt.rcParams['legend.title_fontsize'] = axis_fsize
figsize=(8, 4)
ncols = 4
#if len(sys.argv) > 2:
#    cluster_width = float(sys.argv[2])  # Adjust as needed
#else:
#    cluster_width = 0.03
my_cmap = plt.get_cmap("viridis")

def aoi_to_label(aoi):
    if aoi == -1:
        return 'Default'
    else:
        return f'{chr(ord("A") + aoi)}'
    

def legend_wrapper(ax):
    return ax.legend(title='Application Policy', ncol=ncols, loc=2, framealpha=0.7, fontsize=tick_fsize, labelspacing=0.1, handletextpad=0.1, columnspacing=0.4)#bbox_to_anchor=(0.95, 1))  # Legend outside the plot

# Set up argument parser
parser = argparse.ArgumentParser(description='Process input file and figure directory.')

parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
parser.add_argument('fig_dir', type=str, nargs='?', default='../figures/migration/', help='Directory for saving figures (default: ../figures/migration/).')
parser.add_argument('--default_policy', type=str, help='Default policy string.')
args = parser.parse_args()

# Assign arguments to variables
input_file_path = args.input_file
figdir = args.fig_dir

bmark = input_file_path.split('/')[-1].split('.')[0]

# Create figure directory if it doesn't exist
if not os.path.exists(figdir):
    os.makedirs(figdir)
    print(f"Figure directory created: {figdir}")

df = pd.read_csv(input_file_path)

# Group by AOI, Policy, and Problem Size, then calculate the average elapsed time for each group
grouped_df = df.groupby(['Problem Size', 'Policy']).agg({'Kernel Time': 'mean'}).reset_index()
pd.set_option("display.max_rows", None) # show all rows

# Set default policy based on the argument or calculate from the dataframe
if args.default_policy:
    default_policy = list(args.default_policy)
else:
    default_policy = ['m' for i in range(0, len(df['Policy'].unique()[0].split('_')[0]))]  

nallocs = len(default_policy)

df = df[df['Policy'] == "".join(default_policy)]

df['Degree of Subscription'] = df['Problem Size'] * 100 // 12


#print(df)

grouped_df = df.groupby(['Degree of Subscription', 'Policy']).agg({'Kernel Time': 'mean'}).reset_index()
grouped_df.loc[grouped_df['Kernel Time']==0,'Kernel Time'] = 10000000 #Mark times that were DNF

time_max = np.max(grouped_df['Kernel Time'])
time_next = np.max(grouped_df[grouped_df['Kernel Time'] != time_max]['Kernel Time'])

default_policy = "".join(default_policy)
fig = plt.figure(figsize=figsize)

plt.plot(grouped_df['Degree of Subscription'], grouped_df['Kernel Time'])
#lgd = legend_wrapper(fig.get_axes()[0])


# Set plot properties
plt.xlabel('Degree of Subscription', fontsize=axis_fsize)
plt.ylabel('Kernel Time(s)', fontsize=axis_fsize)
#Show the plot
plt.savefig(f"{figdir}/{bmark}_migr.png")#, bbox_extra_artists=(lgd,), bbox_inches='tight')

exit()









# Create a bar plot for each Policy/Problem Size pair
#print(grouped_df)
i = 0
for policy in grouped_df['Sorted Policy'].unique():
    pol_df = grouped_df[grouped_df['Sorted Policy'] == policy]
    pol_df = pol_df.groupby('Sorted Policy', sort=False)
    for (policy, group) in pol_df:
        
        group['Color'] = [my_cmap(i / len(grouped_df['Sorted Policy'].unique()))] * len(group) #Default color 
        
        #Set OOM bars
        group.loc[group['Speedup']<=0.01,'Color'] = 'red' #Angry color for OOM bar 
        group.loc[group['Speedup']<=0.01,'Speedup'] = 0.5
        #plt.bar(positions + i * cluster_width, mean_values, color=my_cmap(nallocs * 3.2 + (1/len(grouped_df['Sorted Policy'].unique()))), width=cluster_width, label=label)
    #plt.bar(positions + i * cluster_width, mean_values, color='r', width=cluster_width, label=label)
        def_labeled = True
        #else:
        plt.bar(positions + (i * cluster_width), group['Speedup'], color=group['Color'], width=cluster_width, label=policy, edgecolor='black', linewidth=0.5)
        i = i + 1
plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, [f"{p}" for p in psize_values], fontsize=tick_fsize)
plt.yticks(fontsize=tick_fsize)
#Add legend label for OOM
plt.bar(0, 0, 0, color='red', label='OOM', edgecolor='black', linewidth=0.5)
#Add line for 'mmm'
plt.hlines(1, 0 - cluster_width, 4+(num_clusters * cluster_width), linestyle='-', color='black')

#legend 
if num_clusters > 30:
    lgd = plt.legend(title='Sorted Policy', ncol=4, loc='upper center', mode='expand', framealpha=0.2, fontsize=axis_fsize, labelspacing=0.1, handletextpad=0.1, columnspacing=0.7, bbox_to_anchor=[-0.1, -0.6, 1.2, 0.5])  # Legend outside the plot
    plt.xlabel("")
else:
    lgd = legend_wrapper(fig.get_axes()[0])


# Set plot properties
plt.xlabel('Degree of Subscription', fontsize=axis_fsize)
plt.ylabel('Speedup Relative to mmm', fontsize=axis_fsize)
#Show the plot
plt.savefig(f"{figdir}/{bmark}_mem_press.png")#, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
