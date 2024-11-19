from coral_imports import *
from coral_plotting import *
from coral_run_project import *
import time

# Create Powerpoint Deck
prs = pptx.Presentation('analysis/results/template.pptx')

# Config filepaths
base = os.path.join(os.getcwd(), "analysis", "configs", "base.yaml")
base_float = os.path.join(os.getcwd(), "analysis", "configs", "base_float.yaml")
library = os.path.join(os.getcwd(), "analysis", "library")
weather_fp = os.path.join(os.getcwd(), "analysis", "library", "weather", "vineyard_wind_repr_with_whalesEXTENDED.csv")
weather = pd.read_csv(weather_fp, parse_dates=["datetime"]).set_index("datetime")

# set up yaml reading
def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', 
tuple_constructor)

# Define regions
ne = ['MA','ME','CT','RI','NH','RI/CT']
nynj = ['NY','NJ']
mid = ['NC', 'MD', 'VA', 'DE']

# List all ports
ports = ['salem', 'searsport_1', 'searsport_2', 'new_bedford', 'new_london', 'arthur_kill', 'njwp_1', 'njwp_2', 'sbmt', 'tradepoint_1', 'tradepoint_2', 'portsmouth_1', 'portsmouth_2']
ne_ports = ['salem', 'searsport','new_bedford','new_london']

# Vessel Names and Costs
vessel_types = ['example_wtiv','example_heavy_lift_vessel', 'example_feeder', 'example_heavy_feeder_1kit', 'example_ahts_vessel']
vessel_costs = {
    "example_wtiv": 615,
    "example_heavy_lift_vessel": 625,
    "example_heavy_feeder_1kit": 250,
    "example_feeder": 100,
    "example_ahts_vessel": 80
}

# Parse scenarios names
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
parser.add_argument('scenarios', nargs='+')
args = parser.parse_args()

filename = args.filename
results_fp = 'analysis/results/%s' % filename
savename = os.path.join(results_fp, '%s.pptx' % filename)

os.makedirs(results_fp)

scenarios = args.scenarios
# print(scenarios)

dfs = []
all_alloc = []
all_future = []



for s in scenarios:

    with open('analysis/scenarios/%s.yaml' % s) as f:
        scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)

    p = os.path.join(os.getcwd(), "analysis", "pipelines", "%s.csv" % scenario['pipeline'])

    start_time = time.time()
    pipeline = Pipeline(p, base, base_float, ffiv_feeders=True)

    description = scenario['description']

    slide = add_text_slide(prs, s, description)
    allocations = scenario['allocations']
    future_resources = scenario['future_resources']
    future_remove = scenario['future_remove']

    coral_time = time.time()
    manager, df = run_manager(pipeline, allocations, library, weather, future_resources=future_resources, future_remove=future_remove)
    #manager, df = run_manager(pipeline, allocations, library, future_resources=future_resources, future_remove=future_remove)
    print("--- CORAL run time: %s seconds ---" % (time.time() - coral_time))
    all_alloc.append(allocations)
    all_future.append(future_resources)
    dfs.append(df)
    print(df.dtypes)
    run_plots(prs, df, ne_ports)

    df.to_csv(os.path.join(results_fp, '%s.csv' % s), date_format='%Y-%m-%d %H:%M:%S')




df_cap = installed_cap(prs, dfs, scenarios)
# total_invest = vessel_investment_plot(prs, all_alloc, all_future, scenarios, vessel_types, vessel_costs)
# cap_per_investment(prs, df_cap, total_invest)




# Save Powerpoint
prs.save(savename)
print(f'\nresults saved to:\n{savename}')


""" df2 = pd.DataFrame(dfs)
writer= pd.ExcelWriter('analysis/results/4wtivs', engine='xlsxwriter')
df2.to_excel(writer, index=False)
writer.save()
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
print(dfs) """

""" df = pd.DataFrame(manager.logs).iloc[::-1]
df = df.reset_index(drop=True).reset_index()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df['Difference'] = df['Finished']-df['Started']
average_difference = (df['Difference'].mean()/8760) """

#df.to_excel('analysis/results/CalibratedProcessTimes25OverlapUnconstrained.xlsx', index=False)

#print(average_difference)

### Open it

sp.run(f'"{savename}"', shell=True)