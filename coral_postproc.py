from coral_imports import *
from coral_plotting import *

# Create presentation
prs = pptx.Presentation('analysis/results/template.pptx')

# Define regions
ne = ['MA','ME','CT','RI','NH','RI/CT']
nynj = ['NY','NJ']
mid = ['NC', 'MD', 'VA', 'DE']

# List all ports
ports = ['salem', 'searsport_1', 'searsport_2', 'new_bedford', 'new_london_1', 'new_london_2', 'arthur_kill', 'njwp_1', 'njwp_2', 'sbmt', 'tradepoint_1', 'tradepoint_2', 'portsmouth_1', 'portsmouth_2']
ne_ports = ['salem', 'searsport','new_bedford','new_london']

# Read result path to analyze
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
# parser.add_argument('scenarios', nargs='+')
args = parser.parse_args()

filename = args.filename
results_fp = 'analysis/results/%s' % filename


# Read in dfs from csvs
dfs = []

path = os.path.join(results_fp, '*.csv')

for fname in glob.glob(path):
    df = pd.read_csv(fname, parse_dates=['estimated_cod','Date Initialized','Date Finished', 'Date Started'])
    df = df.drop(df.columns[0],axis=1)
    # pd.to_datetime(df[["estimated_cod"]], unit='ns')
    dfs.append(df)
    print(df.dtypes)

    run_plots(prs, df, ne_ports)



savename = os.path.join(results_fp, 'post_proc.pptx')
prs.save(savename)
print(f'\nresults saved to:\n{savename}')