import sys
sys.path.insert(0, './postprocessing')
from coral_plotting import *

# Create presentation
prs = pptx.Presentation('postprocessing/results/template.pptx')

# List all ports for use in port throughput plots
ports = ['salem', 'searsport', 'new_bedford', 'new_london', 'arthur_kill', 'njwp', 'sbmt', 'tradepoint', 'portsmouth']

# Read result path to analyze
parser = argparse.ArgumentParser("simple_example")
parser.add_argument('filename')
args = parser.parse_args()

filename = args.filename
results_fp = 'postprocessing/results/%s' % filename


# Read in dfs from csvs
path = os.path.join(results_fp, '*.csv')

logs = []
df_investments = []
us_rev = pd.DataFrame(columns=[])
ffiv_rev = pd.DataFrame(columns=[])
wtiv_rev = pd.DataFrame(columns=[])
desc = []

scen_list = list(set([s.replace("_log","").replace("_resource_history","") for s in [os.path.splitext(os.path.basename(fname))[0] for fname in glob.glob(path)]]))

for scenario_name in scen_list:
    log_fname = os.path.join(results_fp, f"{scenario_name}_log.csv")
    history_fname = os.path.join(results_fp, f"{scenario_name}_resource_history.csv")
    log = pd.read_csv(log_fname, parse_dates=['estimated_cod','Date Initialized','Date Finished', 'Date FoundationFinished', 'Date Started'])
    history = pd.read_csv(history_fname)

    summary_table_filename = os.path.join(results_fp, f'{scenario_name}_demand.xlsx')
    df_empty = pd.DataFrame()
    df_empty.to_excel(summary_table_filename, index=False)

    # Extracting the name of the scenario for each csv file and putting it in a new column 
    # such that the corresponding yaml file can be called in coral_plotting
    log['Scenario'] = scenario_name
    desc.append(scenario_name)

    slide = add_text_slide(prs, scenario_name)
 
    log = log.drop(log.columns[0],axis=1)
    
    run_plots(prs, log, history, ports, summary_table_filename)
    # summary_table = percent_resource_demand(history_to_print, summary_table_filename)

    wb = load_workbook(summary_table_filename)
    del wb['Sheet1']
    wb.save(summary_table_filename)

    # with pd.ExcelWriter(summary_table_filename, engine='openpyxl', mode='a') as writer:
    #     summary_table.to_excel(writer, sheet_name=scenario_name)

    # summary_table.to_excel(os.path.join(results_fp, 'percent_count.xlsx'), sheet_name=scenario_name)
    # history_to_print.to_csv(os.path.join(results_fp, 'history', f'{scenario_name}_history.csv'))

    logs.append(log) 

slide = add_text_slide(prs, 'Summary Plots', ["Plots comparing runs"])

df_cum = installed_cap(prs,logs,desc)
# compare_installed_cap(prs,logs,desc)


savename = os.path.join(results_fp, '%s_results.pptx' % filename)
prs.save(savename)
print(f'\nresults saved to:\n{savename}')

# wb = load_workbook(summary_table_filename)
# del wb['Sheet1']
# wb.save(summary_table_filename)