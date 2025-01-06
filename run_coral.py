import sys
sys.path.insert(0, './postprocessing')
from coral_helpers import *
import time

# Config filepaths
base = os.path.join(os.getcwd(), "library", "configs", "base.yaml")
base_float = os.path.join(os.getcwd(), "library", "configs", "base_float.yaml")
library = os.path.join(os.getcwd(), "library")
weather_fp = os.path.join(os.getcwd(), "library", "weather", "vineyard_wind_repr_with_whales.csv")
weather = pd.read_csv(weather_fp, parse_dates=["datetime"]).set_index("datetime")

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

# Parse scenarios names
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
parser.add_argument('scenarios', nargs='+')
args = parser.parse_args()

filename = args.filename
results_fp = 'postprocessing/results/%s' % filename
savename = os.path.join(results_fp, '%s.pptx' % filename)
os.makedirs(results_fp)
scenarios = args.scenarios

dfs = []
all_alloc = []
all_future = []

for s in scenarios:

    with open('library/scenarios/%s.yaml' % s) as f:
        scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)

    p = os.path.join(os.getcwd(), "library", "pipelines", "%s.csv" % scenario['pipeline'])

    phase_overlap = scenario.get('phase_overlap',0.2)

    start_time = time.time()
    pipeline = Pipeline(p, base, base_float, phase_overlap, ffiv_feeders=True)

    description = scenario['description']

    allocations = scenario['allocations']
    future_resources = scenario['future_resources']
    future_remove = scenario['future_remove']

    coral_time = time.time()
    manager, df = run_manager(pipeline, allocations, library, weather, future_resources=future_resources, future_remove=future_remove)
    print("--- CORAL run time: %s seconds ---" % (time.time() - coral_time))
    all_alloc.append(allocations)
    all_future.append(future_resources)
    dfs.append(df)

    df.to_csv(os.path.join(results_fp, '%s.csv' % s), date_format='%Y-%m-%d %H:%M:%S')


