from coral_imports import *
from coral_helpers import *


def add_text_slide(prs, title, text, left=0, top=7.2, width=13.33, height=0.3, fontsize=14):
    """Add text slide for scenario description"""
    blank_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(blank_slide_layout)
    slide.shapes.title.text = title

    text_shape = slide.shapes.placeholders[10]

    text_frame = text_shape.text_frame

   
    text_frame.text = text[0]

    for para_str in text[1:]:
        p = text_frame.add_paragraph()
        p.text = para_str

def add_to_pptx(
        prs, title=None, file=None, left=0, top=0.62, width=13.33, height=None,
        verbose=1, slide=None, 
    ):
    """Add current matplotlib figure (or file if specified) to new powerpoint slide"""
    blank_slide_layout = prs.slide_layouts[3]
    if not file:
        image = io.BytesIO()
        plt.savefig(image, bbox_inches = 'tight',format='png')
    else:
        image = file
        if not os.path.exists(image):
            raise FileNotFoundError(image)

    if slide is None:
        slide = prs.slides.add_slide(blank_slide_layout)
        slide.shapes.title.text = title
    slide.shapes.add_picture(
        image,
        left=(None if left is None else Inches(left)),
        top=(None if top is None else Inches(top)),
        width=(None if width is None else Inches(width)),
        height=(None if height is None else Inches(height)),
    )
    if verbose:
        print(title)
    return slide


def add_textbox(
        text, slide,
        left=0, top=7.2, width=13.33, height=0.3,
        fontsize=14,
    ):
    """Add a textbox to the specified slide"""
    textbox = slide.shapes.add_textbox(
        left=(None if left is None else Inches(left)),
        top=(None if top is None else Inches(top)),
        width=(None if width is None else Inches(width)),
        height=(None if height is None else Inches(height)),
    )
    p = textbox.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    font = run.font
    font.size = Pt(fontsize)
    return slide

def full_gantt(prs, df, sorted=False):
    """Gantt chart of full pipeline. Sorted sorts by expected start date."""
    if sorted:
        df = df.drop(columns=['index'])
        df = df.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(df)/4), dpi=200) # LEN(DF)/4
    ax = fig.add_subplot(111)

    bar_color = []
    for i,row in df.iterrows():
        if row['substructure'] == 'monopile':
            bar_color.append("#F0E442")
        elif row['substructure'] == 'gbf':
            bar_color.append("#D55E00")
        elif row['substructure'] == 'jacket':
            bar_color.append("#CC79A7")
        else:
            bar_color.append("#0072B2")

    delay_bar_color = []
    for i,row in df.iterrows():
        if row['substructure'] == 'monopile':
            delay_bar_color.append("#F7F19D")
        elif row['substructure'] == 'gbf':
            delay_bar_color.append("#FFA65F")
        elif row['substructure'] == 'jacket':
            delay_bar_color.append("#E2B2CC")
        else:
            delay_bar_color.append("#77CEFF")
    
    df["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    df["Date Started"].plot(kind="barh", color=delay_bar_color, ax=ax, zorder=4, label="Delay")
    df["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    df.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker=">")
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df['name'])

    mono_delay = matplotlib.patches.Patch(color='#F7F19D', label='Monopile Delay')
    mono_install = matplotlib.patches.Patch(color='#F0E442', label='Monopile Installation')
    gbf_delay = matplotlib.patches.Patch(color='#FFA65F', label='GBF Delay')
    gbf_install = matplotlib.patches.Patch(color='#D55E00', label='GBF Installation')
    jacket_delay = matplotlib.patches.Patch(color='#E2B2CC', label='SBJ Delay')
    jacket_install = matplotlib.patches.Patch(color='#CC79A7', label='SBJ Installation')
    semisub_delay = matplotlib.patches.Patch(color='#77CEFF', label='Semisub Delay')
    semisub_install = matplotlib.patches.Patch(color='#0072B2', label='Semisub Installation')
    ax.legend(handles=[mono_delay, mono_install, gbf_delay, gbf_install, jacket_delay, jacket_install, semisub_delay, semisub_install])

    ax.set_xlim(df["Date Initialized"].min() - dt.timedelta(days=30), df["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted Full Gantt', width=5.25)
    else:
        slide = add_to_pptx(prs,'Full Gantt', width=4.25)
    plt.close(fig)

def regional_gantt(prs, df, region, region_name, sorted=False):
    """Gantt chart of select region pipeline. Region determined by offtake states in region list. 
       Sorted sorts by expected start date."""
    df = df.drop(columns=['index'])
    df_region = df[df['offtake_state'].isin(region)].reset_index(drop=True).reset_index()

    if sorted:
        df_region = df_region.drop(columns=['index'])
        df_region = df_region.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(df_region)/4), dpi=200)
    ax = fig.add_subplot(111)

    bar_color = []
    for _,row in df_region.iterrows():
        if row['substructure'] == 'monopile':
            bar_color.append("#F0E442")
        elif row['substructure'] == 'gbf':
            bar_color.append("#D55E00")
        elif row['substructure'] == 'jacket':
            bar_color.append("#CC79A7")
        else:
            bar_color.append("#0072B2")

    matplotlib.rcParams.update({'hatch.linewidth': 3.0,
                                'hatch.color': 'E8E9EB'})
    
    df_region["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    df_region["Date Started"].plot(kind="barh", color=bar_color, hatch = '//', ax=ax, zorder=4, label="Delay")
    df_region["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    df_region.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Start Date", marker="d")

    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df_region['name'])

    mono = matplotlib.patches.Patch(color='#F0E442', label='Monopile Project Time')
    gbf = matplotlib.patches.Patch(color='#D55E00', label='GBF Project Time')
    jacket = matplotlib.patches.Patch(color='#CC79A7', label='SBJ Project Time')
    semisub = matplotlib.patches.Patch(color='#0072B2', label='Semisub Project Time')
    ax.legend(handles=[mono, gbf, jacket, semisub])


    ax.set_xlim(df["Date Initialized"].min() - dt.timedelta(days=30), df_region["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % region_name)
    else:
        slide = add_to_pptx(prs,'%s Gantt' % region_name)
    plt.close(fig)


def substructure_gantt(prs, df, substructure, sorted=False):
    """ Gantt filtered by either fixed or floating projects. Sorted sorts by expected start date."""

    df = df.drop(columns=['index'])
    if substructure == 'fixed':
        df = df[df['substructure'].isin(["jacket", "monopile"])].reset_index(drop=True).reset_index()
    else:
        df = df[df['depth'] > 200].reset_index(drop=True).reset_index()
    if sorted:
        df = df.drop(columns=['index'])
        df = df.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(df)/4), dpi=200)
    ax = fig.add_subplot(111)

    bar_color = []
    for _,row in df.iterrows():
        if row['us_wtiv']:
            bar_color.append("#F0E442")
        else:
            if row['associated_port'] in ["new_bedford", "sbmt", "tradepoint"]:
                bar_color.append("#50C878")
            else:
                bar_color.append("#0072B2")

    matplotlib.rcParams.update({'hatch.linewidth': 3.0,
                                'hatch.color': 'E8E9EB'})
    
    df["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    df["Date Started"].plot(kind="barh", color=bar_color, hatch = '//', ax=ax, zorder=4, label="Delay")
    df["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    df.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Start Date", marker="d")

    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df['name'])

    ax.legend()

    ax.set_xlim(df["Date Initialized"].min() - dt.timedelta(days=30), df["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % substructure.capitalize())
    else:
        slide = add_to_pptx(prs,'%s Gantt' % substructure.capitalize())
    plt.close(fig)


def port_gantts(prs, df, ports, sorted=False):
    """Gantt chart of specific ports. Creates subplot for each port in ports list. Sorted sorts by expected start date."""
    i = 1
    ports_in_pipeline = df['port'].nunique()
    fig_height = len(df) * (len(ports)/ports_in_pipeline) / 2.5
    fig = plt.figure(figsize=(10, fig_height), dpi=200)
    df_ports = df.drop(columns=['index'])
    num_ports = len(ports)

    for port in ports:
        df_port = df_ports[df_ports['port'] == port].reset_index(drop=True).reset_index()

        if sorted:
            df_port = df_port.drop(columns=['index'])
            df_port = df_port.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

        ax = fig.add_subplot(num_ports,1,i)
    
        bar_color = []
        for _,row in df.iterrows():
            if row['substructure'] == 'monopile':
                bar_color.append("#F0E442")
            elif row['substructure'] == 'gbf':
                bar_color.append("#D55E00")
            elif row['substructure'] == 'jacket':
                bar_color.append("#CC79A7")
            else:
                bar_color.append("#0072B2")

        matplotlib.rcParams.update({'hatch.linewidth': 3.0,
                                    'hatch.color': 'E8E9EB'})
        
        df_port["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
        df_port["Date Started"].plot(kind="barh", color=bar_color, hatch = '//', ax=ax, zorder=4, label="Delay")
        df_port["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

        df_port.plot(kind="scatter", x="Date Initialized", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker="d")

        ax.set_xlabel("")
        ax.set_ylabel("")
        _ = ax.set_yticklabels(df_port['name'])
        ax.set_title(port.capitalize())

        ax.legend()

        ax.set_xlim(df["Date Initialized"].min() - dt.timedelta(days=30), df_port["Date Finished"].max() + dt.timedelta(days=30))

        i += 1

    if sorted:
        slide = add_to_pptx(prs,'Sorted Port Gantts')
    else:
        slide = add_to_pptx(prs,'Port Gantts')    

    plt.close(fig)


def port_throughput(prs, df, region=None):
    if region:
        df = df.drop(columns=['index'])
        df = df[df['location'].isin(region)].reset_index(drop=True).reset_index()
    res = []
    for _, project in df.iterrows():

        if project["Date Finished"].year == project["Date Started"].year:
            res.append((project["Date Finished"].year, project["associated_port"], project["capacity"]))

        else:

            total = project["Date Finished"].date() - project["Date Started"].date()
            for year in np.arange(project["Date Started"].year, project["Date Finished"].year + 1):
                if year == project["Date Started"].year:
                    perc = (dt.date(year + 1, 1, 1) - project["Date Started"].date()) / total

                elif year == project["Date Finished"].year:
                    perc = (project["Date Finished"].date() - dt.date(year, 1, 1)) / total

                else:
                    perc = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)) / total

                res.append((year, project["associated_port"], perc * project["capacity"]))

    throughput = pd.DataFrame(res, columns=["year", "associated_port", "capacity"]).pivot_table(
        index=["year"],
        columns=["associated_port"],
        aggfunc="sum",
        fill_value=0.
    )["capacity"]


    index = np.arange(throughput.index.min(),throughput.index.max()+1)
    throughput = throughput.reindex(index, fill_value=0)
    # throughput = throughput.drop(columns=['associated_port'])

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(111)
    throughput.plot.bar(ax=ax, width=0.75)
    ax.axhline(y=700, color='k', linestyle='--', linewidth=0.8)
    ax.axhline(y=1000, color='k', linestyle='--', linewidth=0.8)

    # mask = (throughput.max(axis=1) >= 700) & (throughput.max(axis=1) <= 1000)
    # ax.fill_between(throughput.index, 1000, 700, where=mask, interpolate=True, alpha=0.8, color='#E6E6FA')

    # Create step plot for shading
    y1 = np.ones(len(throughput.index))*1000
    y2 = np.ones(len(throughput.index))*700 
    # ax.step(throughput.index, [700] * len(throughput), where='mid', linestyle='-', color='none')  # Bottom line

    # Fill between the lines
    # mask = (throughput.max(axis=1) >= 700) & (throughput.max(axis=1) <= 1000)
    ax.axhspan(700, 1000, alpha=0.8, zorder=0, color = '#E6E6FA')

    ax.set_ylim(0, 2500)
    ax.set_ylabel("Annual Capacity Throughput (MW)")
    ax.set_xlabel("")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)

    ax.legend(fontsize=6, ncol=5)

    slide = add_to_pptx(prs,'Port Throughput')
    plt.close(fig)

def us_vessels_built_bar_chart(prs, dfs, desc):
    """Bar chart of the US vessels built in each scenario."""
    vessel_types = ['example_wtiv_us', 'example_ahts_vessel', 'example_feeder']

    name_updates = {
        #'example_wtiv' : 'Foreign WTIV',
        'example_wtiv_us' : 'US WTIV',
        #'example_heavy_lift_vessel' : 'FFIV',
        'example_ahts_vessel' : 'AHTS',
        'example_feeder' : 'Feeder Barge'

    }

    legend_labels = {
        '10_Feeders_No_Foreign_WTIVs': 'Feeder Emphasis (Low Vessel Recruitment)',
        'Hybrid_Optimized_Pipeline_Low': 'Hybrid (Low Vessel Recruitment)',
        '10_Feeders': 'Feeder Emphasis (High Vessel Recruitment)',
        '3_WTIV': 'WTIV Emphasis',
        '6_AHTS': 'AHTS Emphasis',
        '6_AHTS_GBF_Pipeline': 'AHTS Emphasis GBFs',
        'no_action': 'No Action',
        '3_WTIV_No_Foreign_WTIVs': 'WTIV Emphasis (Low Vessel Recruitment)',
        'Hybrid_Optimized_Pipeline': 'Hybrid'


    }

    vessel_counts = {vessel: [] for vessel in vessel_types}

    scenario_path = 'analysis/scenarios'

    for i, df in enumerate(dfs):
        scenario = desc[i]
        scen_yaml = read_yaml(df['Scenario'].iloc[0], scenario_path)
        futures = scen_yaml['future_resources']

        counts = {vessel: 0 for vessel in vessel_types}

        for vessel_type in futures:
            if vessel_type[1] in vessel_types:
                counts[vessel_type[1]] += len(vessel_type[2])

        for vessel in vessel_types:
            vessel_counts[vessel].append(counts[vessel])

    desc = [legend_labels.get(scenario, scenario) for scenario in desc]
    df_vessels_bar = pd.DataFrame(vessel_counts, index=desc)

    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = fig.add_subplot(111)

    #df_vessels_bar.plot(kind='bar', stacked=True, ax=ax)
    #for i, scenario in enumerate(desc):
    df_vessels_bar.plot(kind='bar', stacked=True, ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    name_updates_list = [name_updates.get(label,label) for label in labels]

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Number of US Vessels Built")
    #ax.legend(title="Vessel Type", fontsize=8)
    ax.legend(handles, name_updates_list, fontsize=6)
    ax.set_title("US Vessels Built by Scenario")

    slide = add_to_pptx(prs, 'US Vessels Built')

    return df_vessels_bar

#Write utlization data to csv
""" def vessel_utilization_plot(prs, df):
    fig = plt.figure(figsize=(10, 4), dpi=200)
    ax = fig.add_subplot(111)

    scenario_path = 'analysis/scenarios'
    scen_yaml = read_yaml(df['Scenario'].iloc[0], scenario_path)
    allocs = scen_yaml['allocations']
    futures = scen_yaml['future_resources']
    removals = scen_yaml['future_remove']
    if removals is None:
            removals = []
    df_vessel_util = vessel_hours(df)
    df_vessel_count = vessel_pipeline(allocs, futures, removals)
    df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100

    # Save the DataFrame to a CSV file
    output_dir = 'analysis/results/New_Optimal_Scenarios'
    csv_file_path = os.path.join(output_dir, 'vessel_utilization_data.csv')
    df_perc_util.to_csv(csv_file_path)


    return df_vessel_util / 24 """



""" def vessel_utilization_plot(prs, df):

    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)

    scenario_path = 'analysis/scenarios'
    scen_yaml = read_yaml(df['Scenario'].iloc[0], scenario_path)
    allocs = scen_yaml['allocations']
    futures = scen_yaml['future_resources']
    df_vessel_util = vessel_hours(df)
    # print(df_vessel_util)
    df_vessel_count = vessel_pipeline(allocs,futures)
    df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100
    # print(df_perc_util)
    ax = df_perc_util.plot.bar(rot=90)

    ax.set_xlabel("")
    ax.set_ylabel("Vessel Utilization (%)")
    #ax.set_xticks(range(2025, 2051, 5))
    #ax.set_xticklabels([str(year) for year in range(2025, 2051, 5)])
    ax.legend(fontsize=6)
    ax.set_ylim(0,100)

    slide = add_to_pptx(prs,'Vessel Utilization')
    return(df_vessel_util/24) """

def vessel_utilization_plot(prs, df):
    fig = plt.figure(figsize=(14, 3), dpi=500)
    ax = fig.add_subplot(111)

    scenario_path = 'analysis/scenarios'
    scen_yaml = read_yaml(df['Scenario'].iloc[0], scenario_path)
    allocs = scen_yaml['allocations']
    futures = scen_yaml['future_resources']
    removals = scen_yaml['future_remove']
    if removals is None:
            removals = []
    df_vessel_util = vessel_hours(df)
    df_vessel_count = vessel_pipeline(allocs, futures, removals)
    df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100

    # Generate the line plot
    ax = df_perc_util.plot.line()

    name_updates = {
        'example_wtiv' : 'Foreign WTIV',
        'example_wtiv_us' : 'US WTIV',
        'example_heavy_lift_vessel' : 'FFIV',
        'example_ahts_vessel' : 'AHTS',
        'example_feeder' : 'Feeder Barge'

    }

    handles, labels = ax.get_legend_handles_labels()
    name_updates_list = [name_updates.get(label,label) for label in labels]
    

    ax.set_xlim([2023, 2050])
    ax.set_xticks(range(2025, 2051, 5))
    ax.set_xticklabels([str(year) for year in range(2025, 2051, 5)])
    ax.set_xlabel("")
    ax.set_ylabel("Vessel Utilization (%)")
    ax.legend(handles, name_updates_list, fontsize=6)
    ax.set_ylim(0, 100)

    slide = add_to_pptx(prs, 'Vessel Utilization')
    return(df_vessel_util / 24)

import os

#Currently set up to write to csv file; script to process this code is AvgUtilization.py
def average_vessel_utilization_plot(prs, dfs, desc):
    avg_utilization = {vessel: [] for vessel in ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']}

    for i, df in enumerate(dfs):
        scenario = desc[i]
       

        scenario_path = 'analysis/scenarios'
        scen_yaml = read_yaml(df['Scenario'].iloc[0], scenario_path)
        allocs = scen_yaml['allocations']
        futures = scen_yaml['future_resources']
        removals = scen_yaml['future_remove']
        if removals is None:
            removals = []
       
        df_vessel_util = vessel_hours(df)
        df_vessel_count = vessel_pipeline(allocs, futures, removals)
        df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100

        for vessel in avg_utilization.keys():
            avg_utilization[vessel].append(df_perc_util[vessel].mean())

    df_avg_utilization = pd.DataFrame(avg_utilization, index=desc)

    output_path = 'analysis/results/New_Optimal_Scenarios'
    os.makedirs(output_path, exist_ok=True)
    csv_file_path = os.path.join(output_path, 'average_vessel_utilization.csv')
    df_avg_utilization.to_csv(csv_file_path)

    return avg_utilization


def vessel_revenue_plot(prs, df_vessel_util):

    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)
    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    vessel_rates = []
    df_vessel_cost = df_vessel_util
    rate_path = 'analysis/library/vessels'
    for vessel in vessel_types:
        vessel_yaml = read_yaml(vessel, rate_path)
        vessel_rate = vessel_yaml['vessel_specs']['day_rate']
        df_vessel_cost[vessel] = df_vessel_util[vessel] * vessel_rate / 1e9

    us_revenue = df_vessel_cost[['example_wtiv_us','example_ahts_vessel','example_feeder']].sum(axis=1).cumsum()
    ffiv_revenue = df_vessel_cost[['example_heavy_lift_vessel']].sum(axis=1).cumsum()
    wtiv_revenue = df_vessel_cost[['example_wtiv']].sum(axis=1).cumsum()

    return(us_revenue, ffiv_revenue, wtiv_revenue)


def vessel_investment_plot(prs, desc):
    yrs = np.arange(2023,2043)
    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    vessel_costs = {
        "example_wtiv": 400,
        "example_wtiv_us": 600,
        "example_heavy_lift_vessel": 625,
        "example_feeder": 60,
        "example_ahts_vessel": 175
        }   
    scen_path = 'analysis/scenarios'
    dates = pd.to_datetime(yrs, format='%Y')
    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=200)

    us_investments = pd.DataFrame(index=dates, columns=desc, data=np.zeros((len(yrs), len(desc))))
    total_investments = pd.DataFrame(index=dates, columns=desc, data=np.zeros((len(yrs), len(desc))))
    vessel_counts = []
    for i in range(0,len(desc)):
        scen = read_yaml(desc[i], scen_path)
        alloc = scen['allocations']
        future = scen['future_resources']
        init_alloc = [alloc['wtiv'][1][1], 
                      alloc['wtiv'][2][1], 
                      alloc['wtiv'][0][1], 
                      alloc['ahts_vessel'][0][1], 
                      alloc['feeder'][1][1]]
        vessel_investment = pd.DataFrame(columns=vessel_types, data = np.zeros((len(yrs), len(vessel_types))), index = dates)
        vessel_count = pd.DataFrame(columns=vessel_types, data = np.zeros((len(yrs), len(vessel_types))), index = dates)
        vessel_count.iloc[0] = init_alloc
        # display(vessel_investment)
        for vessel in vessel_types:
            for vessel_type in future:
                if vessel_type[1] == vessel:
                    years = vessel_type[2]
                    # print(vessel_type[1])
                    # print(years)
                    for year in years:
                        vessel_count.loc[[year],vessel] += 1
            vessel_investment[vessel] = vessel_count[vessel] * vessel_costs[vessel]
        
        us_vessels = ['example_feeder', 'example_ahts_vessel', 'example_wtiv_us']
        vessel_investment.loc[:,'us_total'] = vessel_investment[us_vessels].sum(axis=1)
        vessel_investment['us_total'] = vessel_investment['us_total'].cumsum() / 1000
        vessel_investment.loc[:,'total'] = vessel_investment[vessel_types].sum(axis=1)
        vessel_investment['total'] = vessel_investment['total'].cumsum() / 1000
        total_investments[desc[i]] = vessel_investment['total']
        us_investments[desc[i]] = vessel_investment['us_total']

        vessel_count = vessel_count.cumsum()
        vessel_counts.append(vessel_count)

    us_investments['year'] = yrs
    us_investments.set_index('year', inplace=True)

    us_investments['year'] = yrs
    us_investments.set_index('year', inplace=True)
    us_investments.plot(ax=ax)

    ax.set_ylabel('Capital Investment ($B)')
    ax.yaxis.set_major_locator(tck.MaxNLocator(integer=True))
    plt.minorticks_off()
    # plt.tick_params(bottom = False) 
    ax.set_xticks(yrs[::2])
    slide = add_to_pptx(prs, 'Vessel Investment')

    return(us_investments, vessel_counts)


# def invest_summary_plot(prs, us_investments, df_shares):
#     # bar chart with % shares and us investment bar
#     # print(us_investments)
#     fig = plt.figure(figsize=(6,4), dpi=200)
#     ax1 = fig.add_subplot(111)
#     ax2 = fig.add_subplot(121)

#     df_shares.plot.bar(rot=0, ax=ax1, width=0.1, align='edge', stacked=True, secondary_y=True)
#     us_investments.plot.bar(rot=0, ax=ax2, width=-0.1, align='edge', color='tab:green')

#     colors = {'US Investment':'tab:green', 
#               'US Revenue (right)':'tab:blue', 
#               'Foreign Revenue (right)':'tab:orange'}         
#     labels = list(colors.keys())
#     handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
#     # ax.legend(handles, labels, bbox_to_anchor=(0.7,-0.1), fontsize='small')
#     # ax.set_ylabel('Capital Investment in US Vessels ($B)')
#     # ax.set_xticklabels(['US WTIV', 'US Feeder', 'AHTS', 'No Action'])

#     slide = add_to_pptx(prs, 'Vessel Investment Summary')

def invest_per_scenario(prs, us_investments, us_rev, ffiv_rev, wtiv_rev):
    fig, axes = plt.subplots(len(list(us_investments.columns)),1,figsize=(8,10), dpi=200)
    i=0
    titles = ['US WTIV', 'US Feeder', 'Float Out', 'No Action']
    for scenario in list(us_investments.columns):
        us_investments[scenario].plot(ax=axes[i], label='US Investments')
        us_rev[scenario].plot(ax=axes[i],label='US cost to Developers', color = 'tab:green')
        ffiv_rev[scenario].plot(ax=axes[i], label='Foreign-flagged FFIV cost to Developers', color = 'tab:orange')
        wtiv_rev[scenario].plot(ax=axes[i], label='Foreign-flagged WTIV cost to Developers', color = 'tab:red')
        axes[i].set_ylabel('$B')
        axes[i].set_ylim(0,10)
        axes[i].set_xlabel('')
        axes[i].set_title(titles[i])
        i += 1


    axes[0].legend(loc='upper left')
    fig.tight_layout()

    slide = add_to_pptx(prs, 'Vessel Investment Summary')


def invest_w_vessels(prs, us_investments, us_rev, ffiv_rev, wtiv_rev, vessel_counts):
    i=0
    titles = ['US WTIV', 'US Feeder', 'Float Out', 'No Action']
    yrs = pd.to_datetime(pd.Series(np.arange(2023,2046,1)), format='%Y')

    us_rev.drop(us_rev.tail(9).index,inplace = True)
    wtiv_rev.drop(wtiv_rev.tail(9).index,inplace = True)
    ffiv_rev.drop(ffiv_rev.tail(9).index,inplace = True)

    colors = {
        'example_wtiv_us': 'darkgreen',
        'example_feeder': 'forestgreen',
        'example_ahts_vessel': 'limegreen',
        'example_wtiv': 'tab:red',
        'example_heavy_lift_vessel': 'tab:orange'
    }
    vessel_order = ['example_heavy_lift_vessel', 'example_wtiv', 'example_wtiv_us','example_feeder','example_ahts_vessel']

    for scenario in list(us_investments.columns):
        vessel_count = vessel_counts[i]
        vessel_count = vessel_count.reindex(yrs, method='pad')
        vessel_count.index = vessel_count.index.year
        us_investments = us_investments.reindex(us_rev.index, method='pad')
        vessel_count = vessel_count[vessel_order]

        fig, axes = plt.subplots(2,1,figsize=(8,6), dpi=200)
        us_investments[scenario].plot(ax=axes[0], label='US Investments')
        us_rev[scenario].plot(ax=axes[0],label='US cost to Developers', color = 'green')
        ffiv_rev[scenario].plot(ax=axes[0], label='Foreign-flagged FFIV cost to Developers', color = 'tab:orange')
        wtiv_rev[scenario].plot(ax=axes[0], label='Foreign-flagged WTIV cost to Developers', color = 'tab:red')
        axes[0].set_ylabel('$B')
        axes[0].set_ylim(0,10)
        axes[0].set_xlabel('')
        axes[0].set_title(titles[i])
        axes[0].legend(loc='upper left')

        vessel_count.plot.bar(ax=axes[1], width=1, color=colors, stacked=True)
        axes[1].set_ylim(0,16)
        axes[1].set_ylabel('$B')
        axes[1].set_xlabel('')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles[::-1], ['AHTS','Feeder','US-flagged WTIV', 'Foreign-flagged WTIV','FFIV'], loc='upper right')

        fig.tight_layout()
        slide = add_to_pptx(prs, 'Vessel Investment Summary for %s' % titles[i])
        i += 1
        

""" def summary_invest_plot(prs, us_investments, us_rev):
    yrs = pd.to_datetime(pd.Series(np.arange(2023,2046,1)), format='%Y')
    fig, ax = plt.subplots(1,1,figsize=(8,6), dpi=200)

    us_investments = us_investments.reindex(us_rev.index, method='pad')

    # markers = {
    #     'natl_gaps_2us': 'dashed',
    #     'natl_gaps_3foreign': 'dashdot',
    #     'natl_gaps_6AHTS': 'dotted',
    #     'example_no_action': 'solid',
    # }
    linestyle = ['--','-.',':','-']

    colors = ['C0', 'C1', 'C2', 'C3']
    linestyle_investments = '-'
    linestyle_costs = '--'

    us_investments.plot(ax=ax, label='US Investments', color = colors, style=linestyle_investments)
    us_rev.plot(ax=ax,label='US cost to Developers', color = colors, style=linestyle_costs)

    ax.set_ylim(0,5)
    ax.set_ylabel('$ Billion')
    ax.set_xlabel('')
    ax.set_xlim(2025,2045)
    ax.set_xticks(np.arange(2025,2050,5))
    h, l = ax.get_legend_handles_labels()
    h.reverse()
    l = ['US Feeder Emphasis','US WTIV Emphasis','AHTS Emphasis', 'No Action']
    ph = [plt.plot([],marker="", ls="")[0]]*2
    handles = ph[:1] + h[4:] + ph[1:] + h[:4]
    labels = ["US Investment"] + l + ["US Cost to Developer"] + l
    leg = ax.legend(handles, labels, ncol = 2, loc='upper left', fontsize=8)  
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)  
    fig.tight_layout()
    slide = add_to_pptx(prs, 'US Investment Summary') """

def summary_invest_plot(prs, us_investments, us_rev, desc):
    yrs = pd.to_datetime(pd.Series(np.arange(2023, 2046, 1)), format='%Y')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200)

    us_investments = us_investments.reindex(us_rev.index, method='pad')

    colors = [f'C{i % 10}' for i in range(len(desc))]

    for i, scenario in enumerate(desc):
        us_investments[scenario].plot(ax=ax, label=f'US Investments - {scenario}', color=colors[i], linestyle='-')
        us_rev[scenario].plot(ax=ax, label=f'US Cost to Developers - {scenario}', color=colors[i], linestyle='--')

    ax.set_ylim(0, 5)
    ax.set_ylabel('$ Billion')
    ax.set_xlabel('')
    ax.set_xlim(2025, 2045)
    ax.set_xticks(np.arange(2025, 2050, 5))

    handles, labels = ax.get_legend_handles_labels()

    investment_handles = [h for h, l in zip(handles, labels) if 'Investments' in l]
    cost_handles = [h for h, l in zip(handles, labels) if 'Cost to Developers' in l]

    investment_labels = [l.replace('US Investments - ', '') for l in labels if 'Investments' in l]
    cost_labels = [l.replace('US Cost to Developers - ', '') for l in labels if 'Cost to Developers' in l]

    ph = [plt.plot([], marker="", ls="")[0]] * 2 
    combined_handles = ph[:1] + investment_handles + ph[1:] + cost_handles
    combined_labels = ["US Investment"] + investment_labels + ["US Cost to Developer"] + cost_labels
    leg = ax.legend(combined_handles, combined_labels, ncol=2, loc='upper left', fontsize=8)

    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    fig.tight_layout()

    slide = add_to_pptx(prs, 'US Investment Summary')


def compare_investments(prs, df_investments, desc):
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)

    yrs = np.arange(2023,2050,1)
    df_cum_investment = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)
    i=0
    for df in df_investments:
        df.loc[:,'total'] = df.sum(axis=1)
        df_cum_investment[desc[i]] = df['total']
        i+=1

    df_cum_investment.plot(kind='line',ax=ax)
    ax.set_ylabel("Annual Invesment ($M)")
    ax.set_xlabel("")

    slide = add_to_pptx(prs, 'Vessel Investment by Scenario')

    return(df_cum_investment)
    
""" def compare_investments_bar(prs, df_investments, desc):
    fig, ax = plt.subplots(figsize=(14,6), dpi=200)

    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    df_cum_investments_cutoff = {scenario: [] for scenario in desc}
    width = 0.15  

    for i, df in enumerate(df_investments):
        df = df[df.index <= 2040]  # Filter data to include only years up to 2040
        for vessel in vessel_types:
            df_cum_investments_cutoff[desc[i]].append(df[vessel].sum()/1000)

    x = np.arange(len(vessel_types))
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, scenario in enumerate(desc):
        ax.bar(x + i*width, df_cum_investments_cutoff[scenario], width, label=scenario, color=colors[i])

    ax.set_ylabel("Cumulative Costs by 2040 ($B)")
    ax.set_xlabel("Vessel Types")
    ax.set_title('Cumulative Vessel Investment by Vessel Type ')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(vessel_types)
    ax.legend(title="Scenarios")

    slide = add_to_pptx(prs, 'Cumulative Investment by Vessel Type and Scenario (Up to 2040)')


    return df_cum_investments_cutoff """

def compare_investments_bar(prs, us_rev, ffiv_rev, wtiv_rev, desc):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)

    categories = ['Cost to Developers: US-Flagged Vessels', 'Cost to Developers: Foreign-Flagged WTIVs', 'Cost to Developers: Foreign-Flagged FFIVs']
    df_cum_investments_cutoff = {scenario: [] for scenario in desc}
    width = 0.15  # Width of the bars

    for scenario in desc:
        # Directly access the cumulative value at 2040
        us_rev_2040 = us_rev.loc[2040, scenario]
        ffiv_rev_2040 = ffiv_rev.loc[2040, scenario]
        wtiv_rev_2040 = wtiv_rev.loc[2040, scenario]

        # Store the cumulative values for each scenario
        df_cum_investments_cutoff[scenario].extend([us_rev_2040, wtiv_rev_2040, ffiv_rev_2040])

    x = np.arange(len(categories))
    colors = ['C0', 'C1', 'C2', 'C3']

    name_updates = {
        'natl_gaps_no_action_2_1' : 'No Action',
        'natl_gaps_4AHTS_2_1' : 'AHTS Emphasis',
        'natl_gaps_3us_2_1' : 'US WTIV Emphasis',
        'natl_gaps_4foreign_2_1' : 'US Feeder Emphasis'
    }
    
    for i, scenario in enumerate(desc):
        label = name_updates.get(scenario,scenario)
        ax.bar(x + i * width, df_cum_investments_cutoff[scenario], width, label=label, color=colors[i])

    ax.set_ylabel("$B")
    ax.set_title('Cumulative Costs by 2040')
    ax.set_xticks(x + width * (len(desc) - 1) / 2)
    ax.set_xticklabels(categories)
    ax.legend(title="Scenarios")

    fig.tight_layout()

    slide = add_to_pptx(prs, 'Cumulative Investment and Costs by Scenario (Up to 2040)')

    return df_cum_investments_cutoff



def installed_cap(prs, dfs, desc, region = None):
    yrs = np.arange(2023,2065,1)
    df_cap = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)
    df_cum = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)

    df = dfs[0]
    if region:
        df = df.drop(columns=['index'])
        df = df[df['offtake_state'].isin(region)].reset_index(drop=True).reset_index()

    df['cod'] = df['estimated_cod'].dt.year
    df_cod = df.groupby(['cod']).capacity.sum().reset_index()
    df_cod['sum'] = df_cod['capacity'].cumsum(axis=0) / 1000
    # print(df_cod)
    # df_cum['cod'] = df_cod['sum']
    
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(1,1,1)
    #df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

    i=0
    width = 0.25

    for df in dfs:
        df['finished'] = df['Date Finished'].dt.year
        if region:
            df = df.drop(columns=['index'])
            df = df[df['offtake_state'].isin(region)].reset_index(drop=True).reset_index()
        df_finished = df.groupby(['finished']).capacity.sum().reset_index()
        df_finished['capacity'] = df_finished['capacity'] / 1000
        df_finished['sum'] = df_finished['capacity'].cumsum(axis=0)

        cap_mapping = dict(df_finished[['finished', 'capacity']].values)
        df_cap[desc[i]] = df_cap.index.map(cap_mapping).fillna(0)

        df_cum[desc[i]] = df_cap[desc[i]].cumsum(axis=0)
        i += 1

    #order = [2,4,1,3,0]
    #desc = [desc[i]for i in order]

    #colors = ['C3','C0','C2','C4', 'C5','C1']

    #df_cum[desc].plot(linestyle = '-', color=colors, ax=ax, label='cumulative')
    df_cum[desc].plot(linestyle = '-', ax=ax, label='cumulative')
    # df_cap[desc].plot(kind='bar', ax=ax, label='annual')
    ax.set_xlabel("")
    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    cum_label = [s + ' cumulative' for s in desc]
    #cum_label = ["Unconstrained Vessels", "4 US WTIVs Unconstrained Ports", "4 US WTIVs", "3 US WTIVs", "2 US WTIVs", "No Action" ]
    #cum_label = ["6 AHTS Unconstrained Ports", "6 AHTS", "5 AHTS", "4 AHTS", "No Action", "3 AHTS" ]
    #cum_label = ["Unconstrained Vessels", "6 Foreign WTIVs Unconstrained Ports", "6 Foreign WTIVs", "5 Foreign WTIVs", "4 Foreign WTIVs", "3 Foreign WTIVs", "No Action"]
    #cum_label = ["No Action", "No Action, Feedering for Fixed Foundations"]
    #cum_label = ["3 Foreign WTIVs", "3 US WTIVs", "5 AHTS", "No Action"]
    #cum_label = ["Infinite Vessels", "US Feeder Emphasis", "AHTS Emphasis", "US WTIV Emphasis", "No Action"]
    
    #labels = ['COD'] + cum_label
    labels = cum_label
    
    #ax.legend(labels)
    ax.legend(labels, prop={'size': 7})
    #ax.set_ylim([10,80])
    #ax.set_xlim([2022,2043])
    #ax.set_xticks(np.arange(2022,2043,5))
    ax.set_xlim(right=2045)

    slide = add_to_pptx(prs,'Installed Capacity')

    return df_cum

def compare_installed_cap(prs, dfs, desc, region=None):

    df_2040 = pd.DataFrame(columns = ['2040'])
    df_2030 = pd.DataFrame(columns = ['2040'])
    df_2050 = pd.DataFrame(columns = ['2050'])
    
    i=0
    for df in dfs:
        if region:
            df = df.drop(columns=['index'])
            df = df[df['location'].isin(region)].reset_index(drop=True).reset_index()
        cap_by_year = pd.DataFrame()
        cap_by_year['year'] = pd.DatetimeIndex(df['Date Finished']).year
        cap_by_year['capacity'] = df['capacity']
        cap = cap_by_year.groupby(['year'])['capacity'].sum().reset_index()
        cap_2030 = cap.loc[cap['year'] <= 2030]['capacity'].sum()/1e3
        cap_2040 = cap.loc[cap['year'] <= 2040]['capacity'].sum()/1e3
        cap_2050 = cap.loc[cap['year'] <= 2050]['capacity'].sum()/1e3
        row_2030 = {'Scenario': desc[i], '2030': cap_2030}
        row_2040 = {'Scenario': desc[i], '2040': cap_2040}
        row_2050 = {'Scenario': desc[i], '2050': cap_2050}
        df_2030 = df_2030.append(row_2030, ignore_index=True)
        df_2040 = df_2040.append(row_2040, ignore_index=True)
        df_2050 = df_2050.append(row_2050, ignore_index=True)
        i+=1

    df_2040_per_wtiv = df_2040.copy()
    j=1
    for index,row in df_2040.iterrows():
        row['2040'] = row['2040']/j
        df_2040_per_wtiv.iloc[index] = row
        j += 1
    
    df_2040 = df_2040.set_index('Scenario')
    df_2030 = df_2030.set_index('Scenario')
    df_2040_per_wtiv = df_2040_per_wtiv.set_index('Scenario')

    df_caps = pd.DataFrame(index=desc, columns = ['2030','2040','2040_per_wtiv'])
    df_caps['2030'] = df_2030['2030']
    df_caps['2040'] = df_2040['2040']
    df_caps['2040_per_wtiv'] = df_2040_per_wtiv['2040']

    fig = plt.figure(figsize=(6,4), dpi=200)
    ax = fig.add_subplot(111)

    df_caps.plot.bar(rot=0, ax=ax, width=0.3)

    ax.set_ylabel('Installed Capacity (GW)')
    ax.set_xlabel('')
    ax.set_xlim(-0.25,3.25)

    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height() * 1.005), fontsize=6)

    colors = {'2030 Capacity':'tab:blue', 
              '2040 Capacity':'tab:orange',
              '2040 Capacity per # WTIV':'tab:green'}

    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, loc='upper left', prop={'size': 6})
    slide = add_to_pptx(prs,'Summary Installed Cap')

    fig = plt.figure(figsize=(6,4), dpi=200)
    ax = fig.add_subplot(111)
    df_caps = df_caps.transpose()
    df_caps.plot.bar(rot=0, ax=ax, width=0.3)

    ax.set_ylabel('Installed Capacity (GW)')
    ax.set_xlabel('')

    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height() * 1.005), fontsize=6)

    colors = {'1 WTIV':'tab:blue', 
              '2 WTIV':'tab:orange',
              '3 WTIV':'tab:green',
              '4 WTIV':'tab:red'}
     
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, loc='upper left', prop={'size': 6})
    slide = add_to_pptx(prs,'Summary Installed Cap T')

    plt.close()


def installed_cap_region(prs, dfs, desc):
    """Line plots of cumulative installed capacity separated by region."""
    regions = {
        'All Regions': None,
        'NE': ['MA', 'ME', 'CT', 'RI', 'NH', 'RI/CT'],
        'NYNJ': ['NY', 'NJ'],
        'Mid-Atlantic': ['NC', 'MD', 'VA', 'DE']
    }

    yrs = np.arange(2023, 2065, 1)

    for region_name, region_states in regions.items():
        df_cap = pd.DataFrame(columns=desc, data=np.zeros((len(yrs), len(desc))), index=yrs)
        df_cum_region = pd.DataFrame(columns=desc, data=np.zeros((len(yrs), len(desc))), index=yrs)

        df = dfs[0]
        if region_states:
            df = df[df['location'].isin(region_states)].reset_index(drop=True)

        df['cod'] = df['estimated_cod'].dt.year
        df_cod = df.groupby(['cod']).capacity.sum().reset_index()
        df_cod['sum'] = df_cod['capacity'].cumsum(axis=0) / 1000

        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(1, 1, 1)

        #df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

        i = 0
        for df in dfs:
            df['finished'] = df['Date Finished'].dt.year
            if region_states:
                df = df[df['location'].isin(region_states)].reset_index(drop=True)
            df_finished = df.groupby(['finished']).capacity.sum().reset_index()
            df_finished['capacity'] = df_finished['capacity'] / 1000
            df_finished['sum'] = df_finished['capacity'].cumsum(axis=0)

            cap_mapping = dict(df_finished[['finished', 'capacity']].values)
            df_cap[desc[i]] = df_cap.index.map(cap_mapping).fillna(0)
            df_cum_region[desc[i]] = df_cap[desc[i]].cumsum(axis=0)
            i += 1

        df_cum_region[desc].plot(linestyle='-', ax=ax, label='cumulative')
        ax.set_xlabel("")
        ax.set_ylabel("Capacity (GW)")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
        )

        cum_label = [s + ' cumulative' for s in desc]
        #labels = ['Target Pipeline'] + cum_label
        labels = cum_label
        ax.legend(labels, prop={'size': 7})
        ax.set_xlim(right=2045)

        slide_title = f'Installed Capacity - {region_name}'
        slide = add_to_pptx(prs, slide_title)

    return df_cum_region

def installed_cap_state(prs, dfs, desc):
    """Line plots of cumulative installed capacity separated by state, with existing state targets displayed."""
    states = ['MA', 'ME', 'CT', 'RI', 'NH', 'RI/CT', 'NY', 'NJ', 'NC', 'MD', 'VA', 'DE']

    #Some are targets to produce OSW and others are to procure OSW- differentiate?
    state_targets = {
        'MA': (2027, 5.6),
        'ME': (2040, 3),
        'CT': (2030, 2),
        'NY': (2035, 9),
        'NJ': (2035, 7.5),
        'NC': (2030, 2.8),
        'MD': (2031, 8.5),
        'VA': (2034, 5.2)
    }

    yrs = np.arange(2023, 2065, 1)

    for state in states:
        df_cap = pd.DataFrame(columns=desc, data=np.zeros((len(yrs), len(desc))), index=yrs)
        df_cum_state = pd.DataFrame(columns=desc, data=np.zeros((len(yrs), len(desc))), index=yrs)

        df = dfs[0]
        df = df[df['location'] == state].reset_index(drop=True)

        df['cod'] = df['estimated_cod'].dt.year
        df_cod = df.groupby(['cod']).capacity.sum().reset_index()
        df_cod['sum'] = df_cod['capacity'].cumsum(axis=0) / 1000

        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax = fig.add_subplot(1, 1, 1)

        # Plot expected pipeline
        #df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

        i = 0
        for df in dfs:
            df['finished'] = df['Date Finished'].dt.year
            df = df[df['location'] == state].reset_index(drop=True)
            df_finished = df.groupby(['finished']).capacity.sum().reset_index()
            df_finished['capacity'] = df_finished['capacity'] / 1000  # Convert to GW
            df_finished['sum'] = df_finished['capacity'].cumsum(axis=0)

            cap_mapping = dict(df_finished[['finished', 'capacity']].values)
            df_cap[desc[i]] = df_cap.index.map(cap_mapping).fillna(0)
            df_cum_state[desc[i]] = df_cap[desc[i]].cumsum(axis=0)
            i += 1

        df_cum_state[desc].plot(linestyle='-', ax=ax, label='cumulative')

        if state in state_targets:
            target_year, target_capacity = state_targets[state]
            ax.plot(target_year, target_capacity, marker='o', markersize=7, color='red', alpha=.5, label=f'{state} Target')

        ax.set_xlabel("")
        ax.set_ylabel("Capacity (GW)")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
        )

        cum_label = [s + ' cumulative' for s in desc]
        #labels = ['Target Pipeline'] + cum_label
        labels = cum_label
        if state in state_targets:
            labels.append(f'{state} Target')
        ax.legend(labels, prop={'size': 7})
        ax.set_xlim(right=2045)

        slide_title = f'Installed Capacity - {state}'
        slide = add_to_pptx(prs, slide_title)

    return df_cum_state

def installed_cap_target(prs, dfs, desc, region=None):
    """Bar charts of capacity installed by certain years per scenario."""
    target_years = [2035, 2040]
    df_cap_target = pd.DataFrame(index=desc, columns=['2035 Capacity', '2040 Capacity'])

    for i, df in enumerate(dfs):
        if region:
            df = df.drop(columns=['index'])
            df = df[df['offtake_state'].isin(region)].reset_index(drop=True).reset_index()

        df['finished'] = df['Date Finished'].dt.year
        df_finished = df.groupby(['finished']).capacity.sum().reset_index()
        df_finished['capacity'] = df_finished['capacity'] / 1000  # Convert to GW
        df_finished['sum'] = df_finished['capacity'].cumsum(axis=0)

        for year in target_years:
            cumulative_capacity = df_finished[df_finished['finished'] <= year]['capacity'].sum()
            df_cap_target.at[desc[i], f'{year} Capacity'] = cumulative_capacity
   
    df_cap_target = df_cap_target.sort_values(by='2040 Capacity', ascending=False)
    desc = df_cap_target.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

    legend_labels = {
        'natl_gaps_infv': 'Infinite Vessels',
        'natl_gaps_4foreign_2_1': 'US Feeder Emphasis',
        'natl_gaps_4AHTS_2_1': 'AHTS Emphasis',
        'natl_gaps_3us_2_1': 'US WTIV Emphasis',
        'natl_gaps_no_action_2_1': 'No Action'
    }

    bar_width = 0.10
    index = np.arange(2)

    #colors = ['C4', 'C3', 'C1', 'C2', 'C0', 'C5']

    for i, scenario in enumerate(desc):
        ax.bar(index[0] + i * bar_width, df_cap_target.loc[scenario, '2035 Capacity'],
               bar_width, label=legend_labels.get(scenario, scenario), color=f'C{i}')
    for i, scenario in enumerate(desc):
        ax.bar(index[1] + i * bar_width, df_cap_target.loc[scenario, '2040 Capacity'],
               bar_width, color=f'C{i}')
        
    """ for i, scenario in enumerate(desc):
        ax.bar(index[0] + i * bar_width, df_cap_target.loc[scenario, '2035 Capacity'],
               bar_width)
    for i, scenario in enumerate(desc):
        ax.bar(index[1] + i * bar_width, df_cap_target.loc[scenario, '2040 Capacity'],
               bar_width) """
       
    ax.set_xticks(index + 1.5 * bar_width)
    ax.set_xticklabels(['2035', '2040'])

    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    for i, scenario in enumerate(desc):
        ax.text(index[0] + i * bar_width, df_cap_target.loc[scenario, '2035 Capacity'] + 0.1,
                round(df_cap_target.loc[scenario, '2035 Capacity'], 0), ha='center', va='bottom')
        ax.text(index[1] + i * bar_width, df_cap_target.loc[scenario, '2040 Capacity'] + 0.1,
                round(df_cap_target.loc[scenario, '2040 Capacity'], 0), ha='center', va='bottom')

    ax.legend(title="Scenarios", bbox_to_anchor=(1.05, 1), loc='upper left')

    slide = add_to_pptx(prs, 'Cumulative Capacity by 2035 and 2040')

    return df_cap_target

def cap_per_investment(prs, df_cum, df_investments):
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(1,1,1)

    df = df_cum / df_investments
    
    df.plot(ax=ax)
    ax.set_ylabel('Capacity per Investment (MW/$)')
    slide = add_to_pptx(prs, 'Capacity per Investment (MW/$)')

def avg_delay(prs, dfs, desc):
    """Bar chart of the average delay of fixed-bottom projects by scenario. Projects are grouped by their intended COD in the corresponding pipeline csv file for the scenario."""
    cod_groups = {
        '2025-2030': (2025, 2030),
        '2030-2035': (2031, 2035),
        '2035-2040': (2036, 2040),
        #'2040-2045': (2041, 2045)
    }

    """ legend_labels = {
        'natl_gaps_infv': 'Infinite Vessels',
        'natl_gaps_4foreign_2_1': 'US Feeder Emphasis',
        'natl_gaps_4AHTS_2_1': 'AHTS Emphasis',
        'natl_gaps_3us_2_1': 'US WTIV Emphasis',
        'natl_gaps_no_action_2_1': 'No Action'
    } """

    legend_labels = {
    'natl_gaps_2us_2_1': '2 WTIVs',
    'natl_gaps_3us_2_1': '3 WTIVs',
    'natl_gaps_4us_2_1': '4 WTIVs',
    'natl_gaps_no_action_2_1': 'No Action'
}


    regions = {
        'All Regions': None,
        'NE': ['MA', 'ME', 'CT', 'RI', 'NH', 'RI/CT'],
        'NYNJ': ['NY', 'NJ'],
        'Mid-Atlantic': ['NC', 'MD', 'VA', 'DE']
    }


    for region_name, region_states in regions.items():
        df_delay = pd.DataFrame(index=cod_groups.keys(), columns=desc)

        for i, df in enumerate(dfs):
            df['estimated_cod'] = pd.to_datetime(df['estimated_cod'])
            df['Date Started'] = pd.to_datetime(df['Date Started'])
            df['Date Initialized'] = pd.to_datetime(df['Date Initialized'])

            df['delay'] = ((df['Date Started'] - df['Date Initialized']).dt.days) / 365

            #Filter for only fixed-bottom projects
            df = df[df['substructure'].isin(['monopile', 'jacket'])]

            if region_states:
                df = df[df['location'].isin(region_states)]

            for group_name, (start_year, end_year) in cod_groups.items():
                group_df = df[(df['estimated_cod'].dt.year >= start_year) &
                              (df['estimated_cod'].dt.year <= end_year)]
                avg_delay = group_df['delay'].mean() if not group_df.empty else 0
                df_delay.at[group_name, desc[i]] = avg_delay

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bar_width = 0.10 
        index = np.arange(len(cod_groups))

        #order = [0,2,3,1]
        #desc = [desc[i]for i in order]

        for i, scenario in enumerate(desc):
            bars = ax.bar(index + i * bar_width, df_delay[scenario].astype(float),
                   bar_width, label=legend_labels.get(scenario, scenario), color=f'C{i}')
            for bars in ax.containers:
                ax.bar_label(bars, fmt='{:,.1f}')
        


        ax.set_xticks(index + bar_width * (len(desc) - 1) / 2)
        ax.set_xticklabels(cod_groups.keys())

        ax.set_ylabel("Average Delay (Years)")
        ax.set_xlabel("Intended COD")
        ax.legend(title="Scenarios", prop={'size': 8})

        slide_title = f'Average Delay Per Project by COD - {region_name}'
        slide = add_to_pptx(prs, slide_title)

    return df_delay

def cancellations(prs, dfs, desc):
    """Bar chart showing the GW of cancelled projects for each scenario. Projects are grouped by their intended COD."""
    cod_groups = {
        '2025-2030': (2025, 2030),
        '2030-2035': (2031, 2035),
        '2035-2040': (2036, 2040),
        #'2040-2045': (2041, 2045)
    }

    """ legend_labels = {
        'natl_gaps_2us_2_1': '2 WTIVs',
        'natl_gaps_3us_2_1': '3 WTIVs',
        'natl_gaps_4us_2_1': '4 WTIVs',
        'natl_gaps_no_action_2_1': 'No Action'
    } """

    legend_labels = {
        'natl_gaps_infv': 'Infinite Vessels',
        'natl_gaps_4foreign_2_1': 'US Feeder Emphasis',
        'natl_gaps_4AHTS_2_1': 'AHTS Emphasis',
        'natl_gaps_3us_2_1': 'US WTIV Emphasis',
        'natl_gaps_no_action_2_1': 'No Action'
    }

    regions = {
        'All Regions': None,
        'NE': ['MA', 'ME', 'CT', 'RI', 'NH', 'RI/CT'],
        'NYNJ': ['NY', 'NJ'],
        'Mid-Atlantic': ['NC', 'MD', 'VA', 'DE']
    }

    for region_name, region_states in regions.items():
        df_cancel = pd.DataFrame(index=cod_groups.keys(), columns=desc)

        for i, df in enumerate(dfs):
            df['estimated_cod'] = pd.to_datetime(df['estimated_cod'])
            df['Date Started'] = pd.to_datetime(df['Date Started'])
            df['Date Initialized'] = pd.to_datetime(df['Date Initialized'])

            df['delay'] = ((df['Date Started'] - df['Date Initialized']).dt.days) / 365

            # Filter for only fixed-bottom projects
            df = df[df['substructure'].isin(['monopile', 'jacket'])]

            # Filter for specific regions if region_states is defined
            if region_states:
                df = df[df['location'].isin(region_states)]

            # Filter for projects delayed 2 years or more
            df = df[df['delay'] >= 2]

            for group_name, (start_year, end_year) in cod_groups.items():
                group_df = df[(df['estimated_cod'].dt.year >= start_year) &
                              (df['estimated_cod'].dt.year <= end_year)]

                total_capacity = group_df['capacity'].sum()/1000 if not group_df.empty else 0
                df_cancel.at[group_name, desc[i]] = total_capacity

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bar_width = 0.10
        index = np.arange(len(cod_groups))

        #order = [0, 2, 3, 1]  # Custom order for desc
        #desc = [desc[i] for i in order]

        for i, scenario in enumerate(desc):
            bars = ax.bar(index + i * bar_width, df_cancel[scenario].astype(float),
                          bar_width, label=legend_labels.get(scenario, scenario), color=f'C{i}')
            for bars in ax.containers:
                ax.bar_label(bars, fmt='{:,.1f}')

        ax.set_xticks(index + bar_width * (len(desc) - 1) / 2)
        ax.set_xticklabels(cod_groups.keys())
        ax.set_ylabel("Projects at Risk of Cancellation (GW)")
        ax.set_xlabel("Intended COD")
        ax.legend(title="Scenarios", prop={'size': 8})

        slide_title = f'Total Capacity at Risk by COD - {region_name}'
        slide = add_to_pptx(prs, slide_title)

    return df_cancel


def run_plots(prs, df, ports):
    ne = ['MA','ME','CT','RI','NH','RI/CT']
    nynj = ['NY','NJ']
    mid = ['NC', 'MD', 'VA', 'DE']

    full_gantt(prs, df)
    # full_gantt(prs, df, sorted=True)

    # regional_gantt(prs,, df, ne, 'New England')
    # regional_gantt(prs,, df, ne, 'New England', sorted=True)

    port_throughput(prs,df)
    # port_throughput(prs,df,ne)
    # port_throughput(prs,df,nynj)
    # port_throughput(prs,df,mid)

    # regional_gantt(prs, df, nynj, 'New York/New Jersey')
    # regional_gantt(prs, df, nynj, 'New York/New Jersey', sorted=True)

    # regional_gantt(prs, df, mid, 'Midatlantic')
    # regional_gantt(prs, df, mid, 'Midatlantic', sorted=True)

    # port_gantts(prs, df, ports)
    # port_gantts(prs, df, ports, sorted=True)

    # substructure_gantt(prs, df, 'fixed')
    # substructure_gantt(prs, df, 'fixed', sorted=True)
    # substructure_gantt(prs, df, 'floating')
    # substructure_gantt(prs, df, 'floating', sorted=True)

