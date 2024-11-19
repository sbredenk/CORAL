import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV file
csv_file_path = 'analysis/results/New_Optimal_Scenarios/Utilization/average_vessel_utilization.csv'
df_avg_utilization = pd.read_csv(csv_file_path, index_col=0)


df_avg_utilization.columns = ['Foreign WTIV', 'US WTIV', 'FFIV', 'AHTS', 'Feeder Barge']

df_avg_utilization_transposed = df_avg_utilization.T

# Plotting
fig, ax = plt.subplots(figsize=(14, 6), dpi=200)


df_avg_utilization_transposed.plot(kind='bar', ax=ax, width=0.8)

#ax.set_xlabel("Vessel Types")
ax.set_ylabel("Utilization Rate (%)")
ax.set_title('Average Vessel Utilization by Scenario')
ax.set_xticks(np.arange(len(df_avg_utilization_transposed.index)))
ax.set_xticklabels(df_avg_utilization_transposed.index)
ax.legend(title="Scenarios", fontsize=8)
ax.set_ylim(0,100)


plt.xticks(rotation=0)


plt.tight_layout()


png_file_path = os.path.join(os.path.dirname(csv_file_path), 'average_vessel_utilization_plot.png')
plt.savefig(png_file_path)

