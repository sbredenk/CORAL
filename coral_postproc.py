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

df_investments = []
us_rev = pd.DataFrame(columns=[])
ffiv_rev = pd.DataFrame(columns=[])
wtiv_rev = pd.DataFrame(columns=[])
desc = []

for fname in glob.glob(path):
    df = pd.read_csv(fname, parse_dates=['estimated_cod','Date Initialized','Date Finished', 'Date FoundationFinished', 'Date Started'])
    #Extracting the name of the scenario for each csv file and putting it in a new column such that the corresponding yaml file can be called in coral_plotting
    scenario_name = os.path.splitext(os.path.basename(fname))[0]
    df['Scenario'] = scenario_name
    desc.append(scenario_name)
    slide = add_text_slide(prs, scenario_name, "text")
 
    df = df.drop(df.columns[0],axis=1)
    # pd.to_datetime(df[["estimated_cod"]], unit='ns')
    dfs.append(df) 
    # run_plots(prs, df, ne_ports)

    df_util = vessel_utilization_plot(prs,df)
    us_revenue, ffiv_revenue, wtiv_revenue = vessel_revenue_plot(prs,df_util)

    us_rev[scenario_name] = us_revenue
    ffiv_rev[scenario_name] = ffiv_revenue
    wtiv_rev[scenario_name] = wtiv_revenue

    # df_investments.append(df_investment)


# df_cum_investments = compare_investments(prs, df_investments, desc)
us_investments, vessel_counts = vessel_investment_plot(prs, desc)
# invest_per_scenario(prs, us_investments, us_rev, ffiv_rev, wtiv_rev)
# invest_w_vessels(prs, us_investments, us_rev, ffiv_rev, wtiv_rev, vessel_counts)
summary_invest_plot(prs, us_investments, us_rev)
df_cum = installed_cap(prs,dfs,desc)
# cap_per_investment(prs, df_cum, df_cum_investments)

savename = os.path.join(results_fp, 'test_summary.pptx')
prs.save(savename)
print(f'\nresults saved to:\n{savename}')