import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.cm import ScalarMappable

# Read data from a CSV file
input_file_path = sys.argv[1]
if len(sys.argv) > 2:
    cluster_width = float(sys.argv[2])  # Adjust as needed
else:
    cluster_width = 0.06
my_cmap = plt.get_cmap("tab20b")

df = pd.read_csv(input_file_path, dtype={'Threshold': str})

# Combine 'Policy' and 'Threshold' columns
df['Policy_Threshold'] = df['Policy'] + '_' + df['Threshold'].astype(str)

# Identify rows with AOI == -1
#aoi_minus_one_rows = df[df['AOI'] == -1]

# Duplicate rows for each unique Pressure value and set Elapsed to 0
#new_rows = pd.concat([aoi_minus_one_rows.assign(Pressure=pressure, Elapsed=0) for pressure in df['Pressure'].unique()])

# Concatenate the original DataFrame with the new rows
#df = pd.concat([df, new_rows], ignore_index=True)

for pt in df['Policy_Threshold'].unique():
    for pressure in df['Pressure'].unique():
        if pressure not in df[df['Policy_Threshold'] == pt]['Pressure'].unique():
            existing_pressures = df[df['Policy_Threshold'] == pt]['Pressure'].unique() 
            pressure0rows = df[(df['Policy_Threshold'] == pt) & (df['Pressure'] == min(existing_pressures))]
            if 'h' in pt:
                new_rows = pressure0rows.assign(Pressure=pressure)
            else: 
                new_rows = pressure0rows.assign(Pressure=pressure, Elapsed=0)
            df = pd.concat([df, new_rows], ignore_index=True)

# Group by AOI, Policy_Threshold, and Pressure, then calculate the average elapsed time for each group
grouped_df = df.groupby(['AOI', 'Policy_Threshold', 'Pressure']).agg({'Elapsed': 'mean'}).reset_index()

for pressure in grouped_df['Pressure'].unique():
    working_set = grouped_df[grouped_df['Pressure'] == pressure]
    mmm_time = working_set[working_set['Policy_Threshold'] == 'mmm_50']['Elapsed'].values[0]
    best_line = working_set[working_set['Elapsed'] == min(working_set[working_set['Elapsed'] > 0]['Elapsed'])]
    best_strat = best_line['Policy_Threshold'].values[0]
    best_time = best_line['Elapsed'].values[0]
    if float(mmm_time) > 0: 
        improvement = (float(mmm_time) - float(best_time)) / float(mmm_time)
        print(f"Pressure {pressure}\nmmm: {mmm_time}\n{best_strat}: {best_time}\nImprovement: {improvement}")

# Plot each unique AOI value separately
for aoi_value in grouped_df['AOI'].unique():
    if aoi_value != -1:
        #subset_df = grouped_df[grouped_df['AOI'] == aoi_value]
        subset_df = pd.concat([grouped_df[grouped_df['AOI'] == aoi_value], grouped_df[grouped_df['AOI'] == -1]], ignore_index=True)

        # Set up positions for each bar cluster
        pressure_values = np.unique(subset_df['Pressure'])
        num_clusters = len(subset_df['Policy_Threshold'].unique())
        positions = np.arange(len(pressure_values)) + (cluster_width * (num_clusters - 1) / 2)

        # Create a bar plot for each Policy_Threshold/Pressure pair
        plt.figure(figsize=(10, 6))
        for i, (policy_threshold, group) in enumerate(subset_df.groupby('Policy_Threshold')):
            mean_values = group['Elapsed']
            #plt.bar(positions + i * cluster_width, mean_values, color=my_cmap(i / len(subset_df['Policy_Threshold'].unique())), width=cluster_width, label=policy_threshold)
            plt.bar(positions + i * cluster_width, mean_values, width=cluster_width, label=policy_threshold)
        # Set plot properties
        plt.title(f'Average Elapsed Time for AOI {aoi_value}')
        plt.xlabel('Pressure')
        plt.ylabel('Average Elapsed Time')
        plt.xticks(positions + ((num_clusters - 1) / 2) * cluster_width, pressure_values)
        plt.legend(title='Policy_Threshold', bbox_to_anchor=(0.95, 1))  # Legend outside the plot

        # Show the plot
        plt.savefig(f"clustered_bar_plot_AOI_{aoi_value}.pdf", bbox_inches='tight')

plt.show()

