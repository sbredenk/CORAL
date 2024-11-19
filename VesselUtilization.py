import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'analysis/results/New_Optimal_Scenarios/Utilization/vessel_utilization_data.csv'
df = pd.read_csv(csv_file_path, index_col=0)

# Plotting
plt.figure(figsize=(14, 8), dpi=200)
ax = plt.gca()

# Plot a bar for each vessel on each year
df.plot(kind='bar', ax=ax, width=0.8)

# Set the y-axis from 0 to 100
ax.set_ylim(0,100)
#ax.set_xlim(2023,2050)

# Customize x-axis
ax.set_xlabel("Year")

# Set x-ticks and labels for specific years
years = df.index.astype(int)
xticks = [2025, 2030, 2035, 2040, 2045, 2050]
xtick_positions = [years.get_loc(year) for year in xticks if year in years]

# Apply xticks only for the specified years
ax.set_xticks(xtick_positions)
ax.set_xticklabels([str(year) for year in xticks if year in years])
#ax.set_xticks(range(len(df.index)))
#ax.set_xticklabels(df.index, rotation=0)

#ax.set_xticks(range(2025, 2051, 5))
#ax.set_xticklabels([str(year) for year in range(2025, 2051, 5)])

# Customize y-axis
ax.set_ylabel("Percent Utilization (%)")
ax.set_title("Vessel Utilization: No Action Case")

# Legend settings
ax.legend(title="Vessels", fontsize=10, loc='upper right')

# Save the plot as a PNG file
output_dir = 'analysis/results/New_Optimal_Scenarios/Utilization'
png_file_path = f"{output_dir}/vessel_utilization_over_time.png"
plt.tight_layout()
plt.savefig(png_file_path)

plt.show()