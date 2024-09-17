from coral_imports import *
import time

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

# run manager function 
def run_manager(pipeline, allocations, library, weather=None, future_resources=None, sorted=False):
    manager = GlobalManager(pipeline.configs, allocations, weather, library_path=library)

    if future_resources != None: 
        for i in future_resources:
            manager.add_future_resources(i[0], i[1], i[2])
        
    manager.run()

    # Format DataFrame for figure building
    df = pd.DataFrame(manager.logs).iloc[::-1]
    df = df.reset_index(drop=True).reset_index()

    df_cols = ['substructure','depth', 'location','associated_port', 'capacity','us_wtiv']

    for col in df_cols:
        map = pipeline.projects[["name", col]].set_index("name").to_dict()[col]
        df[col] = [map[name] for name in df['name']]
    
    cod_map = pipeline.projects[["name", "estimated_cod"]].set_index("name").to_dict()['estimated_cod']
    df['estimated_cod'] = [cod_map[name] for name in df['name']]
    df['estimated_cod'] = pd.to_datetime(df['estimated_cod'], format='%Y')

    return manager, df

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

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

    allocations = scenario['allocations']
    future_resources = scenario['future_resources']

    coral_time = time.time()
    manager, df = run_manager(pipeline, allocations, library, weather, future_resources=future_resources)
    print("--- CORAL run time: %s seconds ---" % (time.time() - coral_time))
    all_alloc.append(allocations)
    all_future.append(future_resources)
    dfs.append(df)

    df.to_csv(os.path.join(results_fp, '%s.csv' % s), date_format='%Y-%m-%d %H:%M:%S')


