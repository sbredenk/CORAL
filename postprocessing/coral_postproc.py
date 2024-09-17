from coral_imports import *
from coral_plotting import *

# Create presentation
prs = pptx.Presentation('analysis/results/template.pptx')

# List all ports for use in port throughput plots
ports = ['salem', 'searsport', 'new_bedford', 'new_london', 'arthur_kill', 'njwp', 'sbmt', 'tradepoint', 'portsmouth']

# Read result path to analyze
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
args = parser.parse_args()

filename = args.filename
results_fp = 'analysis/results/%s' % filename


# Read in dfs from csvs
path = os.path.join(results_fp, '*.csv')

dfs = []
df_investments = []
us_rev = pd.DataFrame(columns=[])
ffiv_rev = pd.DataFrame(columns=[])
wtiv_rev = pd.DataFrame(columns=[])
desc = []

for fname in glob.glob(path):
    df = pd.read_csv(fname, parse_dates=['estimated_cod','Date Initialized','Date Finished', 'Date FoundationFinished', 'Date Started'])

    # Extracting the name of the scenario for each csv file and putting it in a new column 
    # such that the corresponding yaml file can be called in coral_plotting
    scenario_name = os.path.splitext(os.path.basename(fname))[0]
    df['Scenario'] = scenario_name
    desc.append(scenario_name)
    slide = add_text_slide(prs, scenario_name, "text")
 
    df = df.drop(df.columns[0],axis=1)
    
    run_plots(prs, df, ports)

    dfs.append(df) 



df_cum = installed_cap(prs,dfs,desc)
compare_installed_cap(prs,dfs,desc)


savename = os.path.join(results_fp, 'test_summary.pptx')
prs.save(savename)
print(f'\nresults saved to:\n{savename}')