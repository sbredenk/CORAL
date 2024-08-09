from coral_imports import *
from coral_plotting import *
from coral_run_project import *

# Create Powerpoint Deck
prs = pptx.Presentation('analysis/results/template.pptx')
savename = 'analysis/results/base_extra_float_port.pptx'

# Config filepaths
base = os.path.join(os.getcwd(), "analysis", "configs", "base.yaml")
base_float = os.path.join(os.getcwd(), "analysis", "configs", "base_float.yaml")
library = os.path.join(os.getcwd(), "analysis", "library")

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
ports = ['searsport', 'new_bedford', 'new_london', 'njwp', 'sbmt', 'tradepoint', 'portsmouth']
ne_ports = ['searsport','new_bedford','new_london']


### BASELINE PIPELINE ###

## BEST GUESS PATHWAY ##

with open('analysis/scenarios/base_baseline.yaml') as f:
    base_scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)

base_baseline = os.path.join(os.getcwd(), "analysis", "pipelines", "%s.csv" % base_scenario['pipeline'])
pipeline = Pipeline(base_baseline, base, base_float, enforce_feeders=True)

description = [
    'Pipeline: Baseline',
    'Vessel Pathway: Base',
    'Port Pathway: Base',
]
slide = add_text_slide(prs, 'Baseline/Best Guess', description)

# Define vessel and port allocations
allocations = base_scenario['allocations']


# Define future resource dates
future_resources = base_scenario['future_resources']

# Configure CORAL, add future resources and run
base_manager, base_df = run_manager(pipeline, allocations, library, future_resources)

# Plot Gantt Charts

run_plots(prs, base_manager, base_df)


## UNCONSTRAINED VESSEL PATHWAY ##

description = [
    'Pipeline: Baseline',
    'Vessel Pathway: Unconstrained',
    'Port Pathway: Base'
]
slide = add_text_slide(prs, 'Basline/Unconstrained Vessels', description)

# Define vessel and port allocations
allocations = {
    "wtiv": [('example_heavy_lift_vessel', 100),('example_wtiv', 100)],
    "feeder": ('example_heavy_feeder', 100),
    "port": [('new_london', 1), ('new_bedford', 1), ('sbmt', 0), ('njwp', 0), ('searsport', 1), ('tradepoint', 0), ('portsmouth', 0), ('salem', 1)],
    "ahts_vessel": ('example_ahts_vessel', 100),
    "towing_vessel": ('example_towing_vessel', 100),
}


# Define future resource dates
future_resources = [
    ['wtiv','example_wtiv', [dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2026,1,1)]],
    ['feeder', 'example_heavy_feeder', [dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1)]],
    ['port','sbmt', [dt.date(2027,1,1)]],
    ['port','njwp', [dt.date(2025,1,1), dt.date(2029,1,1)]],
    ['port','tradepoint',[dt.date(2025,1,1)]],
    ['port','portsmouth',[dt.date(2024,1,1)]],
    ['port','searsport', [dt.date(2025,1,1)]]
                    ]


# Configure CORAL, add future resources and run
infv_manager, infv_df = run_manager(pipeline, allocations, library, future_resources)

# Plot Gantt Charts

run_plots(prs, infv_manager, infv_df)



## UNCONSTRAINED PORT PATHWAY ##

description = [
    'Pipeline: Baseline',
    'Vessel Pathway: Base',
    'Port Pathway: Unconstrained'
]
slide = add_text_slide(prs, 'Basline/Unconstrained Vessels', description)

# Define vessel and port allocations
allocations = {
    "wtiv": [('example_heavy_lift_vessel', 3),('example_wtiv', 2)],
    "feeder": ('example_heavy_feeder', 8),
    "port": [('new_london', 100), ('new_bedford', 100), ('sbmt', 100), ('njwp', 100), ('searsport', 100), ('tradepoint', 100), ('portsmouth', 100), ('salem', 100)],
    "ahts_vessel": ('example_ahts_vessel', 2),
    "towing_vessel": ('example_towing_vessel', 2),
}


# Define future resource dates
future_resources = [
    ['wtiv','example_wtiv', [dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2026,1,1)]],
    ['feeder', 'example_heavy_feeder', [dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1), dt.date(2025,1,1)]],
    ['port','sbmt', [dt.date(2027,1,1)]],
    ['port','njwp', [dt.date(2025,1,1), dt.date(2029,1,1)]],
    ['port','tradepoint',[dt.date(2025,1,1)]],
    ['port','portsmouth',[dt.date(2024,1,1)]],
    ['port','searsport', [dt.date(2025,1,1)]]
                    ]


# Configure CORAL, add future resources and run
infp_manager, infp_df = run_manager(pipeline, allocations, library, future_resources)

# Plot Gantt Charts

run_plots(prs, infp_manager, infp_df)


### COMPARISON PLOTS ###
description = [
    "Base Pathway",
    "Unconstrained Vessels",
    "Unconstraied Ports"
]
slide = add_text_slide(prs, 'Comparison Plots', description)
installed_cap(prs, [base_df, infv_df, infp_df], ['Base Pathway', 'Unconstrained Vessels', 'Unconstrained Ports'])


# Save Powerpoint
prs.save(savename)
print(f'\nresults saved to:\n{savename}')

### Open it
sp.run(f'"{savename}"', shell=True)






## UNCONSTRAINED VESSELS ##