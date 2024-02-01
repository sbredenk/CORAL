import os
import pandas as pd
import numpy as np

from CORAL import FloatingPipeline, GlobalManager
from CORAL.utils import get_installed_capacity_by, get_action_logs
from ORBIT.core.library import initialize_library
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
initialize_library("library")

from helpers import allocations, future_allocations, target_capacity
from plot_routines import plot_gantt, plot_throughput, plot_gantt_nt, assign_colors, plot_summary, plot_deployment, plot_investments, plot_per_dollar, plot_new_gantt, plot_total_investments, plot_deployment2

# Configure scenarios and keep_inputs
projects = "library/pipeline/wc-pipeline.xlsx"
scenarios = ['55 GW']
# scenarios = ['25 GW (SC)', '25 GW (CC)', '35 GW', '55 GW']
base = "base.yaml"
library_path = "library"
weather_path = "library/weather/humboldt_weather_2010_2018.csv"
inv_path = 'library/investments/scenario-investments.xlsx'
# Weather data used by ORBIT for Northern CA:
#weather_path = "library/weather/northern_CA_swh_150m.csv"

weather_year = 2011
weather_on = True

savedir = "results"

# O&M port activities (6 week window)
OM_start_date = datetime(weather_year, 6, 1, 00, 00, 00)
OM_end_date = datetime(weather_year, 7, 15, 00, 00, 00)

capacity_2045=[]
cap_dir = 'results/cumulative-capacity.xlsx'
writer = pd.ExcelWriter(cap_dir)

if __name__ == '__main__':

    weather = pd.read_csv(weather_path, parse_dates=["datetime"]).set_index("datetime")
    # Extract weather from preferred year
    weather_year = weather.iloc[weather.index.year == weather_year]
    # Define column for when the O&M port is in use - True for when O&M activiteis take place
    weather_year['port_in_use'] = False
    weather_year.loc[(weather_year.index > OM_start_date) & (weather_year.index < OM_end_date), "port_in_use"] = True

    if weather_on == True:
        weather_long = pd.concat([weather_year]*50) # Need a 50+ year time series for limited port scenario (should be the longest)
    else:
        weather_long = None

    for s in scenarios:
        pipeline = FloatingPipeline(projects, base, sheet_name=s)
        manager = GlobalManager(pipeline.configs, allocations[s], library_path=library_path, weather=weather_long)

        # Check (and add) any port or vessel resources that are not online at the start of the simulation
        for s_fa,v in future_allocations.items():
            if s_fa == s:
                for vi in v:
                    manager.add_future_resources(vi[0], vi[1], vi[2])

        manager.run()

        # Output action logs and group them by vessel and activity type.
        dfs=[]

        for project in manager._projects.items():
            data = pd.DataFrame.from_dict(project[1].actions)
            dfs.append(data)

        actions_df = pd.concat(dfs)

        actions_filename = str(s) + '_actions_log.csv'
        actions_df.to_csv(savedir + '/Actions/Ungrouped_action_logs/' + actions_filename)

        # Group action log by agent and action
        agents_df = actions_df.groupby(['agent','phase', 'action']).sum(numeric_only=True)['duration']

        agents_filename = str(s) + '_agent_actions_sum.csv'
        agents_df.to_csv(savedir + '/Actions/Action_logs_for_emissions/' + agents_filename)

        # Plot and save results, assign ports to projects
        df = pd.DataFrame(manager.logs).iloc[::-1]
        df = df.reset_index(drop=True).reset_index()

        port_map = pipeline.projects[["name", "associated_port", "capacity"]].set_index("name").to_dict()['associated_port']
        df['port'] = [port_map[name] for name in df['name']]

        region_map = pipeline.projects[["name", "reference_site_region"]].set_index("name").to_dict()['reference_site_region']
        df['region'] = [region_map[name] for name in df['name']]

        capacity_map = pipeline.projects[["name", "capacity"]].set_index("name").to_dict()['capacity']
        df['capacity'] = [capacity_map[name] for name in df['name']]

        # savefig = savedir + '/s' + '_gantt'
        filename = 'Full-Scenario-Gantt/' + str(s) + '_gantt'
        savefig = os.path.join(os.getcwd(), savedir, filename)

        # color_by = "region" or "port"
        color_by = "port"
        assign_colors(df, color_by)
        #plot_gantt(df, manager, s, color_by, fname=savefig)
        inv_df = pd.read_excel(inv_path, sheet_name='schedule')
        plot_new_gantt(df, manager, s, color_by, inv_df, fname=savefig)

        # Plot first five projects:
        first_projs = 5
        filename_nt = 'Near-term-Gantt/' + str(s) + '_nt_gantt'
        savefig_nt = os.path.join(os.getcwd(), savedir, filename_nt)
        plot_gantt_nt(df, manager, first_projs, color_by, fname=savefig_nt)

        # create a .csv file with cumulative installed capacities
        df['finish-year'] = pd.DatetimeIndex(df['Date Finished']).year

        minyear = df['finish-year'].min()
        maxyear = df['finish-year'].max()
        all_years = list(range(minyear, maxyear+1))
        annual_cap = []
        for year in all_years:
            installed_capacity = get_installed_capacity_by(df, year)
            annual_cap.append(installed_capacity)
        caps = pd.DataFrame(list(zip(all_years, annual_cap)), columns =['Year', 'Cumulative Capacity'])
        caps.to_excel(writer, sheet_name=str(s), index=False)
#        c = pd.concat([c, caps], axis=1)
#        c.to_csv('results/all-capacities.csv')

        capacity_2045.append((get_installed_capacity_by(df, 2045))/1000)

        # Annual throughput
        res = []

        for _, project in df.iterrows():

            if project["Date Finished"].year == project["Date Started"].year:
                res.append((project["Date Finished"].year, project["port"], project["capacity"]))

            else:

                total = project["Date Finished"].date() - project["Date Started"].date()
                for year in np.arange(project["Date Started"].year, project["Date Finished"].year + 1):
                    if year == project["Date Started"].year:
                        perc = (dt.date(year + 1, 1, 1) - project["Date Started"].date()) / total

                    elif year == project["Date Finished"].year:
                        perc = (project["Date Finished"].date() - dt.date(year, 1, 1)) / total

                    else:
                        perc = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)) / total

                    res.append((year, project["port"], perc * project["capacity"]))

        throughput = pd.DataFrame(res, columns=["year", "port", "capacity"]).pivot_table(
            index=["year"],
            columns=["port"],
            aggfunc="sum",
            fill_value=0.
        )["capacity"]

        # Plot throughput
        filename_thp = 'Throughput/' + str(s) + '_throughput'
        savefig = os.path.join(os.getcwd(), savedir, filename_thp)
        plot_throughput(throughput, fname=savefig)

        # Save the project dataframe
#        csv_name = 'results/' + s + '_data.csv'
#        df.to_csv(csv_name)

writer.close()

inv_fig = 'library/investments/total-investments.xlsx'
plot_total_investments(inv_fig)
plot_deployment()
# plot_deployment2()
percent_installed = plot_summary(scenarios, capacity_2045, target_capacity)
print(percent_installed)
plot_per_dollar(scenarios, percent_installed, target_capacity)
