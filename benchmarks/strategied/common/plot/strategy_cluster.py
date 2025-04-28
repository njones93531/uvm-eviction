import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.cm import ScalarMappable

        

def groups(aoi, default):
    default = list(default)
    h = default.copy()
    h[aoi] = 'h'
    d = default.copy()
    d[aoi] = 'd'
    return ["".join(x) for x in [h, d]]

def get_aois(default):
    ret = []
    for i, c in enumerate(default):
        if c == 'm':
            ret.append(i)
    return ret


figsize=(8, 4)
ncols = 7
figdir = '../figures/mempress/'
# Read data from a CSV file
parser = argparse.ArgumentParser(description='Parse two data files and create a side by side bar chart of chosen strategies')
parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='The text files to be parsed.')
parser.add_argument('-f1', '--file1', help='input file 1', required=True)
parser.add_argument('-s1', '--strats1', help='Strats for file 1', required=True)
parser.add_argument('-l1', '--labels1', help='Legend labels for strats in file 1')
parser.add_argument('-p1', '--psizes1', help='Problem sizes for file 1', required=True)
parser.add_argument('-f2', '--file2', help='input file 2', required=True)
parser.add_argument('-s2', '--strats2', help='Strats for file 2', required=True)
parser.add_argument('-l2', '--labels2', help='Legend labels for strats in file 2')
parser.add_argument('-p2', '--psizes2', help='Problem sizes for file 2', required=True)
args = parser.parse_args()


input_file_path, psizes, policies, default_policy = parse_args()
bmark = input_file_path.split('/')[-1].split('.')[0]
#if len(sys.argv) > 2:
#    cluster_width = float(sys.argv[2])  # Adjust as needed
#else:
#    cluster_width = 0.03
my_cmap = plt.get_cmap("viridis")

def legend_wrapper(ax):
    return ax.legend(title='Policy', ncol=ncols, loc=2, framealpha=0.7, fontsize=8, labelspacing=0.1, handletextpad=0.1, columnspacing=0.4)#bbox_to_anchor=(0.95, 1))  # Legend outside the plot

df = pd.read_csv(input_file_path)

# Group by AOI, Policy, and Problem Size, then calculate the average elapsed time for each group
grouped_df = df.groupby(['Problem Size', 'Policy']).agg({'Kernel Time': 'mean'}).reset_index()
pd.set_option("display.max_rows", None) # show all rows

if default_policy == []:
    default_policy = ['m' for i in range(0, len(df['Policy'].unique()[0].split('_')[0]))]

nallocs = len(default_policy)
#print(f"Default policy: {default_policy}")
#for psize in grouped_df['Problem Size'].unique():
#    working_set = grouped_df[grouped_df['Problem Size'] == psize]
#    if len(working_set[working_set['Policy'] == default_policy]['Kernel Time']) == 0:
#        default_time = 7200.0
#    else
#        default_time = working_set[working_set['Policy'] == default_policy]['Kernel Time'].values[0]
#    best_line = working_set[working_set['Kernel Time'] == min(working_set[working_set['Kernel Time'] > 0]['Kernel Time'])]
#    best_strat = best_line['Policy'].values[0]
#    best_time = best_line['Kernel Time'].values[0]
#    print(f"Problem Size {psize}\ndefault:\t\t{'{:10.4f}'.format(default_time)}\n{best_strat}:    \t\t{'{:10.4f}'.format(best_time)}")
#    if float(default_time) != 0: 
#        improvement = (float(default_time) - float(best_time)) / float(default_time) * 100.0
#        print(f"Improvement:\t\t{'{:10.2f}'.format(improvement)}%")
#        speedup = (float(default_time) / float(best_time))
#        print(f"Speedup:\t\t{'{:10.2f}'.format(speedup)}x")
    

df = df[df['Problem Size'].isin(psizes)]
df.loc[:, "aoi"] = -1
for aoi in get_aois(default_policy):
    df.loc[df['Policy'].isin(policies), "aoi"] = aoi
    for psize in psizes:
        new_row = df[(df['Policy'] == "".join(default_policy)) & (df['Problem Size'] == psize)].iloc[0].copy()
        new_row["aoi"] = aoi
        ##print(new_row)
        df = df.append(new_row)

#print(df)

grouped_df = df.groupby(['Problem Size', 'aoi', 'Policy']).agg({'Kernel Time': 'mean'}).reset_index()
#grouped_df = grouped_df[grouped_df['Kernel Time'] > 0]
grouped_df = grouped_df[grouped_df["aoi"] > -1]
grouped_df.reset_index().sort_values("aoi")

#print("Success")

# Set up positions for each bar cluster
num_clusters = len(policies)
cluster_width = 0.8 / num_clusters
positions = np.arange(len(psizes)) + (cluster_width * (num_clusters) / 2)
time_max = np.max(grouped_df['Kernel Time'])
time_next = np.max(grouped_df[grouped_df['Kernel Time'] != time_max]['Kernel Time'])

flip = False

if False:
  print("oops")
    # time_max > 10 * time_next:
   # f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

   # # plot the same data on both axes
   # grouped_df = grouped_df.groupby('Policy', sort=False)
   # for i, (policy, group) in enumerate(grouped_df):
   #     if flip:
   #         mean_values = group['Kernel Time'].iloc[::-1]
   #     else:
   #         mean_values = group['Kernel Time']

   #     ax.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)
   #     ax2.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)

   # # zoom-in / limit the view to different portions of the data
   # ax.set_ylim(time_max - (time_next // 2), (time_max + (time_next // 2)))  # outliers only
   # ax2.set_ylim(0, 1.2 * time_next)  # most of the data
   # if time_max == 7200:
   #     ax.hlines(7200, 0, 1.2, color='r', linestyle='--', label='Cutoff')

   # # hide the spines between ax and ax2
   # ax.spines['bottom'].set_visible(False)
   # ax2.spines['top'].set_visible(False)
   # ax.xaxis.tick_top()
   # ax.tick_params(labeltop=False)  # don't put tick labels at the top
   # plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, [f"{p} GB" for p in psizes])
   # #ax.tick_params(
   # #    axis='x',          # changes apply to the x-axis
   # #    which='both',      # both major and minor ticks are affected
   # #    bottom=False,      # ticks along the bottom edge are off
   # #    top=False,         # ticks along the top edge are off
   # #    labelbottom=False) # labels along the bottom edge are off
   # # This looks pretty good, and was fairly painless, but you can get that
   # # cut-out diagonal lines look with just a bit more work. The important
   # # thing to know here is that in axes coordinates, which are always
   # # between 0-1, spine endpoints are at these locations (0,0), (0,1),
   # # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
   # # appropriate corners of each of our axes, and so long as we use the
   # # right transform and disable clipping.

   # d = .015  # how big to make the diagonal lines in axes coordinates
   # # arguments to pass to plot, just so we don't keep repeating them
   # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
   # ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
   # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

   # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
   # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
   # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal   
   # lgd = legend_wrapper(ax)

   # #ax.set_title('Average Kernel Time (s)')
   # #plt.xlabel('Problem Size')
   # #plt.ylabel('Average Kernel Time (s)')
else:
    def_labeled = False
    default_policy = "".join(default_policy)
    fig = plt.figure(figsize=figsize)
    # Create a bar plot for each Policy/Problem Size pair
    #print(grouped_df)
    i = 0 
    for aoi in grouped_df['aoi'].unique():
        aoi_df = grouped_df[grouped_df['aoi'] == aoi]
        aoi_df = aoi_df.groupby('Policy', sort=False)
        for (policy, group) in aoi_df:
            if flip:
                mean_values = group['Kernel Time'].iloc[::-1]
            else:
                mean_values = group['Kernel Time']
            if policy == default_policy:
                if not def_labeled: 
                    label = policy
                else:
                    label = ""
                #plt.bar(positions + i * cluster_width, mean_values, color=my_cmap(nallocs * 3.2 + (1/len(grouped_df['Policy'].unique()))), width=cluster_width, label=label)
                plt.bar(positions + i  * cluster_width, mean_values, color='r', width=cluster_width, label=label)
                def_labeled = True
            else:
                plt.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)
            i = i + 1
    if num_clusters > 30:
        lgd = plt.legend(title='Policy', ncol=10, loc='upper center', mode='expand', framealpha=0.2, fontsize=8, labelspacing=0.1, handletextpad=0.1, columnspacing=0.7, bbox_to_anchor=[-0.1, -0.6, 1.2, 0.5])  # Legend outside the plot
        plt.xlabel("")
    else:
        lgd = legend_wrapper(fig.get_axes()[0])
    plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, [f"{p}" for p in psizes])
    if time_max == 7200:
        plt.hlines(7200, positions[0], positions[-1] + (num_clusters * cluster_width), color='b', linestyle='--', label='Cutoff')
    if time_max >= 36000:
        plt.hlines(36000, positions[0], positions[-1] + (num_clusters * cluster_width), color='b', linestyle='--', label='Cutoff')
           #plt.bar(positions + i * cluster_width, mean_values, width=cluster_width, label=policy)
# Set plot properties
plt.xlabel('Problem Size (GB)')
plt.ylabel('Average Kernel Time (s)')
#Show the plot
plt.savefig(f"{figdir}/{bmark}_strat_cluster.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
