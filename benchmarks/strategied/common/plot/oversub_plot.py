import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.cm import ScalarMappable

figsize=(8, 4)
ncols = 6
figdir = '../figures/performance'
# Read data from a CSV file
input_file_path = sys.argv[1]
bmark = sys.argv[1].split('/')[-1].split('.')[0]
#if len(sys.argv) > 2:
#    cluster_width = float(sys.argv[2])  # Adjust as needed
#else:
#    cluster_width = 0.03
my_cmap = plt.get_cmap("viridis")

def legend_wrapper(ax):
    return ax.legend(title='Policy', ncol=ncols, loc=2, framealpha=0.2, fontsize=8, labelspacing=0.1, handletextpad=0.1, columnspacing=0.4)#bbox_to_anchor=(0.95, 1))  # Legend outside the plot

df = pd.read_csv(input_file_path)

# Group by AOI, Policy, and Problem Size, then calculate the average elapsed time for each group
grouped_df0 = df.groupby(['Problem Size', 'Policy']).agg({'Kernel Time': 'mean'}).reset_index()
pd.set_option("display.max_rows", None) # show all rows

default_policy = ['m' for i in range(0, len(df['Policy'].unique()[0].split('_')[0]))]
print(f"Default policy: {default_policy}")

for psize in grouped_df0['Problem Size'].unique():
#    working_set = grouped_df[grouped_df['Problem Size'] == psize]
#    if len(working_set[working_set['Policy'] == default_policy]['Kernel Time']) == 0:
#        default_time = 7200.0
#    else:
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
    

    grouped_df = grouped_df0[grouped_df0['Kernel Time'] > 0]
    grouped_df_psize = grouped_df[grouped_df['Problem Size'] == psize].sort_values('Kernel Time')
    grouped_df_16 = grouped_df[grouped_df['Problem Size'] == 16.72].sort_values('Kernel Time')
#grouped_df_9 = grouped_df[grouped_df['Problem Size'] == 9]
#grouped_df = pd.concat([grouped_df_psize, grouped_df_16, grouped_df]).drop_duplicates(keep=False).sort_values('Problem Size')
    flip = False
    if len(grouped_df_psize) > 0:
        flip = True
        grouped_df = grouped_df_psize.reset_index().sort_values('Kernel Time')
    if len(grouped_df_16) > 0:
        grouped_df = grouped_df_16.reset_index().sort_values('Kernel Time')


#Print top 5 policies
    top_policies = grouped_df.sort_values(by='Kernel Time').head(5)
    top_policies['Sorted Policy'] = top_policies['Policy'].apply(lambda x: ''.join(sorted(x)))

# Print the policies with the lowest kernel times
    print(f"{psize}, top polcies: {top_policies[['Sorted Policy', 'Kernel Time']]}")


# Set up positions for each bar cluster
    psize_values = np.unique(grouped_df['Problem Size'])
    num_clusters = len(grouped_df['Policy'].unique())
    cluster_width = 0.8 / num_clusters
    positions = np.arange(len(psize_values)) + (cluster_width * (num_clusters - 1) / 2)
    time_max = np.max(grouped_df['Kernel Time'])
    time_next = np.max(grouped_df[grouped_df['Kernel Time'] != time_max]['Kernel Time'])



    if time_max > 10 * time_next:
        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

        # plot the same data on both axes
        grouped_df = grouped_df.groupby('Policy', sort=False)
        for i, (policy, group) in enumerate(grouped_df):
            if flip:
                mean_values = group['Kernel Time'].iloc[::-1]
            else:
                mean_values = group['Kernel Time']

            ax.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)
            ax2.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)

        # zoom-in / limit the view to different portions of the data
        ax.set_ylim(time_max - (time_next // 2), (time_max + (time_next // 2)))  # outliers only
        ax2.set_ylim(0, 1.2 * time_next)  # most of the data
        if time_max == 7200:
            ax.hlines(7200, 0, 1.2, color='r', linestyle='--', label='Cutoff')
        if time_max == 36000:
            ax.hlines(36000, 0, 1.2, color='r', linestyle='--', label='Cutoff')


        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, [f"{p}" for p in psize_values])
        #ax.tick_params(
        #    axis='x',          # changes apply to the x-axis
        #    which='both',      # both major and minor ticks are affected
        #    bottom=False,      # ticks along the bottom edge are off
        #    top=False,         # ticks along the top edge are off
        #    labelbottom=False) # labels along the bottom edge are off
        # This looks pretty good, and was fairly painless, but you can get that
        # cut-out diagonal lines look with just a bit more work. The important
        # thing to know here is that in axes coordinates, which are always
        # between 0-1, spine endpoints are at these locations (0,0), (0,1),
        # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
        # appropriate corners of each of our axes, and so long as we use the
        # right transform and disable clipping.

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal   
        lgd = legend_wrapper(ax)

        #ax.set_title('Average Kernel Time (s)')
        #plt.xlabel('Problem Size')
        #plt.ylabel('Average Kernel Time (s)')
    else:
        fig = plt.figure(figsize=figsize)
        # Create a bar plot for each Policy/Problem Size pair
        grouped_df = grouped_df.groupby('Policy', sort=False)
        for i, (policy, group) in enumerate(grouped_df):
            if flip:
                mean_values = group['Kernel Time'].iloc[::-1]
            else:
                mean_values = group['Kernel Time']
            plt.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(grouped_df['Policy'].unique())), width=cluster_width, label=policy)
        if num_clusters > 30:
            lgd = plt.legend(title='Policy', ncol=10, loc='upper center', mode='expand', framealpha=0.2, fontsize=8, labelspacing=0.1, handletextpad=0.1, columnspacing=0.7, bbox_to_anchor=[-0.1, -0.6, 1.2, 0.5])  # Legend outside the plot
            plt.xlabel("")
        else:
            lgd = legend_wrapper(fig.get_axes()[0])
        plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, [f"{p}" for p in psize_values])
        if time_max == 7200:
            plt.hlines(7200, 0.4, 1.2, color='r', linestyle='--', label='Cutoff')
        #           #plt.bar(positions + i * cluster_width, mean_values, width=cluster_width, label=policy)
# Set plot properties
    plt.xlabel('Problem Size (GB)')
    plt.ylabel('Average Kernel Time (s)')
#Show the plot
    plt.savefig(f"{figdir}/{bmark}_{psize}_perf.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
