import pandas as pd

file_path = 'analysis/results/optimal_scenarios_2_1/natl_gaps_infv.csv'
df = pd.read_csv(file_path)

df['Date TurbineStart'] = pd.to_datetime(df['Date TurbineStart'])
df['Date Finished'] = pd.to_datetime(df['Date Finished'])


df_filtered = df[df['substructure'].isin(['monopile', 'jacket'])]


events = []

# Create events for the start and end of each project
for _, row in df_filtered.iterrows():
    events.append((row['Date TurbineStart'], 'start'))
    events.append((row['Date Finished'], 'end'))

# Sort events by date, 'start' events first if the dates are the same
events.sort(key=lambda x: (x[0], x[1] == 'end'))


max_wtivs = 0
current_wtivs = 0

# Calculate the number of overlapping projects
for event in events:
    if event[1] == 'start':
        current_wtivs += 1
        max_wtivs = max(max_wtivs, current_wtivs)
    elif event[1] == 'end':
        current_wtivs -= 1

# Output the maximum number of WTIVs being used simultaneously
print(f"Maximum number of WTIVs utilized at the same time: {max_wtivs}")



df['Date Started'] = pd.to_datetime(df['Date Started'])


df_filtered2 = df[df['substructure'].isin(['semisub', 'gbf'])]


events2 = []

# Create events for the start and end of each project
for _, row in df_filtered2.iterrows():
    events2.append((row['Date Started'], 'start'))
    events2.append((row['Date Finished'], 'end'))

# Sort events by date, 'start' events first if the dates are the same
events2.sort(key=lambda x: (x[0], x[1] == 'end'))


max_AHTS = 0
current_AHTS = 0

# Calculate the number of overlapping projects
for event2 in events2:
    if event2[1] == 'start':
        current_AHTS += 1
        max_AHTS = max(max_AHTS, current_AHTS)
    elif event2[1] == 'end':
        current_AHTS -= 1

# Output the maximum number of AHTS vessels being used simultaneously
print(f"Maximum number of AHTS vessels utilized at the same time: {max_AHTS}")