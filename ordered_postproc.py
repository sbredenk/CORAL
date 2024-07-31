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
parser.add_argument('scenarios', nargs='+')
args = parser.parse_args()

filename = args.filename
results_fp = 'analysis/results/%s' % filename


# Read in dfs from csvs
dfs = []

path = os.path.join(results_fp, '*.csv')

df_revenues = []
desc = []


for s in args.scenarios:
    df = pd.read_csv('%s/%s.csv' % (results_fp, s), parse_dates=['estimated_cod','Date Initialized','Date Finished', 'Date FoundationFinished', 'Date Started'])
    #Extracting the name of the scenario for each csv file and putting it in a new column such that the corresponding yaml file can be called in coral_plotting
    scenario_name = os.path.splitext(os.path.basename('%s/%s.csv' % (results_fp, s)))[0]
    df['Scenario'] = scenario_name
    desc.append(scenario_name)

    df = df.drop(df.columns[0],axis=1)
    # pd.to_datetime(df[["estimated_cod"]], unit='ns')
    dfs.append(df) 
    # run_plots(prs, df, ne_ports)

    # df_util = vessel_utilization_plot(prs,df)
    # df_revenue = vessel_revenue_plot(prs,df_util)
    # df_revenues.append(df_revenue)


# df_cum_revenues = compare_revenues(prs, df_revenues, desc)
df_cum = installed_cap(prs,dfs,desc,region=ne)
us_invest = vessel_investment_plot(prs, dfs, desc)
df_out = vessel_port_invest(us_invest, desc)
summary_cap(prs, dfs, desc, df_out, region=ne)
# cap_per_investment(prs, df_cum, df_cum_investments)

savename = os.path.join(results_fp, 'lim_ports.pptx')
prs.save(savename)
print(f'\nresults saved to:\n{savename}')