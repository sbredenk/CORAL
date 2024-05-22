from coral_imports import *
from coral_plotting import *
from coral_run_project import *

# Create Powerpoint Deck
prs = pptx.Presentation('analysis/results/template.pptx')

# Config filepaths
base = os.path.join(os.getcwd(), "analysis", "configs", "base.yaml")
base_float = os.path.join(os.getcwd(), "analysis", "configs", "base_float.yaml")
library = os.path.join(os.getcwd(), "analysis", "library")
weather_fp = os.path.join(os.getcwd(), "analysis", "library", "weather", "vineyard_wind_repr_with_whales.csv")

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
ports = ['salem', 'searsport', 'new_bedford', 'new_london', 'njwp', 'sbmt', 'tradepoint', 'portsmouth']
ne_ports = ['salem', 'searsport','new_bedford','new_london']

# Vessel Names and Costs
vessel_types = ['example_wtiv','example_heavy_lift_vessel','example_ahts_vessel']
vessel_costs = {
    "example_wtiv": 615,
    "example_heavy_lift_vessel": 625,
    "example_ahts_vessel": 80
}

# Parse scenarios names
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
parser.add_argument('scenarios', nargs='+')
args = parser.parse_args()

filename = args.filename
savename = 'analysis/results/%s.pptx' % filename

scenarios = args.scenarios
# print(scenarios)

dfs = []
all_alloc = []
all_future = []

for s in scenarios:
    with open('analysis/scenarios/%s.yaml' % s) as f:
        scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)

    p = os.path.join(os.getcwd(), "analysis", "pipelines", "%s.csv" % scenario['pipeline'])
    pipeline = Pipeline(p, base, base_float, enforce_feeders=True)

    description = scenario['description']

    slide = add_text_slide(prs, s, description)
    allocations = scenario['allocations']
    future_resources = scenario['future_resources']

    manager, df = run_manager(pipeline, allocations, weather, library, future_resources)
    all_alloc.append(allocations)
    all_future.append(future_resources)
    dfs.append(df)
    run_plots(prs, manager, df, ne_ports)



df_cap = installed_cap(prs, dfs, scenarios, ne)
df_investments = vessel_investment_plot(prs, all_alloc, all_future, scenarios, vessel_types, vessel_costs)
cap_per_investment(prs, df_cap, df_investments)

# Save Powerpoint
prs.save(savename)
print(f'\nresults saved to:\n{savename}')

### Open it
sp.run(f'"{savename}"', shell=True)