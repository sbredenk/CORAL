__author__ = "Sophie Bredenkamp"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Sophie Bredenkamp"
__email__ = "sophie.bredenkamp@nrel.gov"

from coral_imports import *
from coral_helpers import *


def add_text_slide(prs, title, left=0, top=7.2, width=13.33, height=0.3, fontsize=14):
    """Add text slide for scenario description"""
    blank_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(blank_slide_layout)
    slide.shapes[0].text = title

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

def plot_shared_resource_capacities(prs, history, ignore_cols=None, col_map=None):
    """
    Plots number of shared resources in poolin each year.

    Parameters
    ----------
    history: DataFrame
        Shared resource history.
    ignore_cols : list (optional)
        List of columns to ignore.
    col_map : dict (optional)
        Map of column name to label.    
    """

    if ignore_cols is None:
        ignore_cols = []

    if col_map is None:
        col_map = {}

    history['time'] = pd.to_datetime(history['time'])
    history["time"] = history["time"].dt.date
    history = history.groupby("time").tail(1)
    

    cols = [
        c for c in list(history.columns) if c not in ["time", *ignore_cols]
    ]


    def invert(column):
        return max(column) - column
    
    history[cols] = history[cols].apply(invert)

    vessel_cols = [c for c in cols if "port" not in c]
    port_cols = [c for c in cols if "port" in c]


    fig,axs = plt.subplots(len(vessel_cols)-1, 1, sharey=True, figsize=(10, 20))
    fig.tight_layout()
    for i, col in enumerate(vessel_cols[1:]):

        data = list(zip(history["time"], history[col]))
        data = squarify(data)

        try:
            label = col_map[col]

        except KeyError:
            label = col

        axs[i].plot(data["time"], data["capacity"], color='g')
        axs[i].set_title(label)
        axs[i].set_ylabel("Resource Capacity")
        axs[i].set_ylim([0,6])
        axs[i].set_xlim([dt.date(2020, 1, 1), dt.date(2055, 1, 1)])

    slide = add_to_pptx(prs,'Shared Resource Capacity - Vessels', width=4.25)

    fig,axs = plt.subplots(len(port_cols), 1, sharey=True, figsize=(10, 20))
    fig.tight_layout()
    for i, col in enumerate(port_cols):

        data = list(zip(history["time"], history[col]))
        data = squarify(data)

        try:
            label = col_map[col]

        except KeyError:
            label = col

        axs[i].plot(data["time"], data["capacity"], color='g')
        axs[i].set_title(label)
        axs[i].set_ylabel("Resource Capacity")
        axs[i].set_ylim([0,6])
        axs[i].set_xlim([dt.date(2020, 1, 1), dt.date(2055, 1, 1)])

    slide = add_to_pptx(prs,'Shared Resource Capacity - Ports', width=4.25)

    history = history.drop(ignore_cols, axis=1)

    return history

def percent_resource_demand(history, filename):

    history.set_index('time', inplace=True)
    new_time_index = pd.date_range(start=history.index.min(), end=history.index.max(), freq='1D')
    history_resampled = history.reindex(new_time_index)
    history_resampled.fillna(method='ffill', inplace=True)
    # history_resampled.reset_index(inplace=True)
    # history_resampled.reset_index(drop=True)
    history_resampled = history_resampled.drop(['Unnamed: 0'],axis=1)

    decades = [(2025, 2030), (2030, 2035), (2035,2040), (2040, 2045), (2045, 2100)]



    for d_low, d_high in decades:
        summary_table = pd.DataFrame(columns=history_resampled.columns)
        history_in_decade = history_resampled[(history_resampled.index.year >= d_low) & (history_resampled.index.year < d_high)]
        if not history_in_decade.empty:
            for i in range(0,int(history_in_decade.max().max())+1):
                new_row = {}
                for col in history_in_decade.columns:
                    value = (history_in_decade[col] == i).sum() / len(history_in_decade)
                    new_row[col] = value
                summary_table = summary_table.append(new_row, ignore_index=True)

            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                summary_table.to_excel(writer, sheet_name=f'{d_low}-{d_high}')

            # Step 2: Apply percentage formatting using openpyxl directly
            wb = load_workbook(filename)
            ws = wb[f'{d_low}-{d_high}']

            # Format the 'Completion' column (assumed to be column B)
            for row in ws.iter_rows(min_row=2, min_col=2):
                for cell in row:
                    cell.number_format = '0.0%'

            # Save changes
            wb.save(filename)
    return summary_table

def full_gantt(prs, log, sorted=False):
    """
    Gantt chart of full pipeline.

    Parameters
    ----------
    prs : object
        Powerpoint presentation
    log : DataFrame
        CORAL run log
    sorted : bool (optional)
        Sorts projects by expected start date
    """
    if sorted:
        log = log.drop(columns=['index'])
        log = log.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(log)/4), dpi=200)
    ax = fig.add_subplot(111)

    bar_color = []
    for _,row in log.iterrows():
        if row['substructure'] == 'monopile':
            bar_color.append("#F0E442")
        elif row['substructure'] == 'gbf':
            bar_color.append("#D55E00")
        elif row['substructure'] == 'jacket':
            bar_color.append("#CC79A7")
        else:
            bar_color.append("#0072B2")

    delay_bar_color = []
    for _,row in log.iterrows():
        if row['substructure'] == 'monopile':
            delay_bar_color.append("#F7F19D")
        elif row['substructure'] == 'gbf':
            delay_bar_color.append("#FFA65F")
        elif row['substructure'] == 'jacket':
            delay_bar_color.append("#E2B2CC")
        else:
            delay_bar_color.append("#77CEFF")
    
    log["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    log["Date Started"].plot(kind="barh", color=delay_bar_color, ax=ax, zorder=4, label="Delay")
    log["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    log.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker=">")
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(log['name'])

    mono_delay = matplotlib.patches.Patch(color='#F7F19D', label='Monopile Delay')
    mono_install = matplotlib.patches.Patch(color='#F0E442', label='Monopile Installation')
    gbf_delay = matplotlib.patches.Patch(color='#FFA65F', label='GBF Delay')
    gbf_install = matplotlib.patches.Patch(color='#D55E00', label='GBF Installation')
    jacket_delay = matplotlib.patches.Patch(color='#E2B2CC', label='SBJ Delay')
    jacket_install = matplotlib.patches.Patch(color='#CC79A7', label='SBJ Installation')
    semisub_delay = matplotlib.patches.Patch(color='#77CEFF', label='Semisub Delay')
    semisub_install = matplotlib.patches.Patch(color='#0072B2', label='Semisub Installation')
    ax.legend(handles=[mono_delay, mono_install, gbf_delay, gbf_install, jacket_delay, jacket_install, semisub_delay, semisub_install])

    ax.set_xlim(log["Date Initialized"].min() - dt.timedelta(days=30), log["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted Full Gantt', width=5.25)
    else:
        slide = add_to_pptx(prs,'Full Gantt', width=4.25)

    plt.close(fig)

def regional_gantt(prs, log, region, region_name, sorted=False):
    """
    Gantt chart of regional pipeline.

    Parameters
    ----------
    prs : object
        Powerpoint presentation
    log : DataFrame
        CORAL run log
    region : list
        States in region of interest
    region_name : str
        Label for region
    sorted : bool (optional)
        Sorts projects by expected start date
    """
    log = log.drop(columns=['index'])
    df_region = log[log['location'].isin(region)].reset_index(drop=True).reset_index()

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

    delay_bar_color = []
    for _,row in df_region.iterrows():
        if row['substructure'] == 'monopile':
            delay_bar_color.append("#F7F19D")
        elif row['substructure'] == 'gbf':
            delay_bar_color.append("#FFA65F")
        elif row['substructure'] == 'jacket':
            delay_bar_color.append("#E2B2CC")
        else:
            delay_bar_color.append("#77CEFF")
    
    df_region["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    df_region["Date Started"].plot(kind="barh", color=delay_bar_color, ax=ax, zorder=4, label="Delay")
    df_region["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    df_region.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker=">")
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df_region['name'])

    mono_delay = matplotlib.patches.Patch(color='#F7F19D', label='Monopile Delay')
    mono_install = matplotlib.patches.Patch(color='#F0E442', label='Monopile Installation')
    gbf_delay = matplotlib.patches.Patch(color='#FFA65F', label='GBF Delay')
    gbf_install = matplotlib.patches.Patch(color='#D55E00', label='GBF Installation')
    jacket_delay = matplotlib.patches.Patch(color='#E2B2CC', label='SBJ Delay')
    jacket_install = matplotlib.patches.Patch(color='#CC79A7', label='SBJ Installation')
    semisub_delay = matplotlib.patches.Patch(color='#77CEFF', label='Semisub Delay')
    semisub_install = matplotlib.patches.Patch(color='#0072B2', label='Semisub Installation')
    ax.legend(handles=[mono_delay, mono_install, gbf_delay, gbf_install, jacket_delay, jacket_install, semisub_delay, semisub_install])

    ax.set_xlim(df_region["Date Initialized"].min() - dt.timedelta(days=30), df_region["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % region_name)
    else:
        slide = add_to_pptx(prs,'%s Gantt' % region_name)
    plt.close(fig)

def substructure_gantt(prs, log, substructure, sorted=False):
    """
    Gantt chart of substrucutre specific pipeline.

    Parameters
    ----------
    prs : object
        Powerpoint presentation
    log : DataFrame
        CORAL run log
    substructure : str
        Nmae of substructure of interest
    sorted : bool (optional)
        Sorts projects by expected start date
    """

    log = log.drop(columns=['index'])
    if substructure == 'fixed':
        log = log[log['substructure'].isin(["jacket", "monopile"])].reset_index(drop=True).reset_index()
    else:
        log = log[log['depth'] > 200].reset_index(drop=True).reset_index()
    if sorted:
        log = log.drop(columns=['index'])
        log = log.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(log)/4), dpi=200)
    ax = fig.add_subplot(111)
    
    log["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color="#D55E00")
    log["Date Started"].plot(kind="barh", color="#FFA65F", ax=ax, zorder=4, label="Delay")
    log["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    log.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker=">")
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(log['name'])

    delay = matplotlib.patches.Patch(color='#FFA65F', label='Delay')
    install = matplotlib.patches.Patch(color='#D55E00', label='Installation')
    ax.legend(handles=[delay,install])

    ax.set_xlim(log["Date Initialized"].min() - dt.timedelta(days=30), log["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % substructure.capitalize())
    else:
        slide = add_to_pptx(prs,'%s Gantt' % substructure.capitalize())
    plt.close(fig)

def port_gantts(prs, log, ports, sorted=False): 
    """
    Gantt charts of port specific pipelines (subplotted).

    Parameters
    ----------
    prs : object
        Powerpoint presentation
    log : DataFrame
        CORAL run log
    ports : list
        Port names of interest
    sorted : bool (optional)
        Sorts projects by expected start date
    """
    i = 1
    ports_in_pipeline = log['associated_port'].nunique()
    fig_height = len(log) * (len(ports)/ports_in_pipeline) / 2
    fig = plt.figure(figsize=(10, fig_height), dpi=200)
    df_ports = log.drop(columns=['index'])
    num_ports = len(ports)

    for port in ports:
        df_port = df_ports[df_ports['associated_port'] == port].reset_index(drop=True).reset_index()

        if sorted:
            df_port = df_port.drop(columns=['index'])
            df_port = df_port.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

        ax = fig.add_subplot(num_ports,1,i)
    
        bar_color = []
        for _,row in df_port.iterrows():
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
        
        df_port["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color="#D55E00")
        df_port["Date Started"].plot(kind="barh", color="#FFA65F", ax=ax, zorder=4, label="Delay")
        df_port["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

        df_port.plot(kind="scatter", x="Date Started", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker=">")

        ax.set_xlabel("")
        ax.set_ylabel("")
        _ = ax.set_yticklabels(df_port['name'])
        ax.set_title(port.capitalize())

        ax.legend()

        ax.set_xlim(df_port["Date Initialized"].min() - dt.timedelta(days=30), df_port["Date Finished"].max() + dt.timedelta(days=30))
        fig.tight_layout()
        i += 1

    if sorted:
        slide = add_to_pptx(prs,'Sorted Port Gantts')
    else:
        slide = add_to_pptx(prs,'Port Gantts')    

    plt.close(fig)

def port_throughput(prs, log, region=None):
    """
    Plot of port throughput.

    Parameters
    ----------
    prs : object
        Powerpoint presentation
    log : DataFrame
        CORAL run log
    region : list (optional)
        States in region of interest
    """
    if region:
        log = log.drop(columns=['index'])
        log = log[log['location'].isin(region)].reset_index(drop=True).reset_index()
    res = []
    for _, project in log.iterrows():

        if project["Date Finished"].year == project["Date Started"].year:
            res._append((project["Date Finished"].year, project["associated_port"], project["capacity"]))

        else:

            total = project["Date Finished"].date() - project["Date Started"].date()
            for year in np.arange(project["Date Started"].year, project["Date Finished"].year + 1):
                if year == project["Date Started"].year:
                    perc = (dt.date(year + 1, 1, 1) - project["Date Started"].date()) / total

                elif year == project["Date Finished"].year:
                    perc = (project["Date Finished"].date() - dt.date(year, 1, 1)) / total

                else:
                    perc = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)) / total

                res._append((year, project["associated_port"], perc * project["capacity"]))

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

def vessel_utilization_plot(prs, log):

    fig = plt.figure(figsize=(14,4), dpi=500)
    ax = fig.add_subplot(111)

    scenario_path = 'library/scenarios'
    scen_yaml = read_yaml(log['Scenario'].iloc[0], scenario_path)
    allocs = scen_yaml['allocations']
    futures = scen_yaml['future_resources']
    removals = scen_yaml['future_remove']
    if removals is None:
        removals = []
    df_vessel_util = vessel_hours(log)
    df_vessel_count = vessel_pipeline(allocs,futures,removals)
    df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100
    
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

def average_vessel_utilization_plot(prs, logs, desc):
    avg_utilization = {vessel: [] for vessel in ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']}

    for i, log in enumerate(logs):
        scenario = desc[i]

        scenario_path = 'analysis/scenarios'
        scen_yaml = read_yaml(log['Scenario'].iloc[0], scenario_path)
        allocs = scen_yaml['allocations']
        futures = scen_yaml['future_resources']
        removals = scen_yaml['future_remove']
        if removals is None:
            removals = []

        df_vessel_util = vessel_hours(log)
        df_vessel_count = vessel_pipeline(allocs, futures, removals)

        #Defining the start and end years for fixed and floating projects
        fixed_start = log[log['substructure'].isin(['monopile', 'jacket'])]['Date Started'].min().year
        fixed_end = log[log['substructure'].isin(['monopile', 'jacket'])]['Date Finished'].max().year
        floating_start = log[log['substructure'] == 'semisub']['Date Started'].min().year
        floating_end = log[log['substructure'] == 'semisub']['Date Finished'].max().year

        #Can replace fixed_start, fixed_end with specific years
        wtiv_start, wtiv_end = fixed_start, fixed_end
        wtiv_us_start, wtiv_us_end = fixed_start, fixed_end
        heavy_lift_start, heavy_lift_end = fixed_start, fixed_end
        feeder_start, feeder_end = fixed_start, fixed_end
        ahts_start, ahts_end = floating_start, floating_end

        df_vessel_util_wtiv = df_vessel_util.loc[wtiv_start:wtiv_end]
        df_vessel_util_wtiv_us = df_vessel_util.loc[wtiv_us_start:wtiv_us_end]
        df_vessel_util_heavy_lift = df_vessel_util.loc[heavy_lift_start:heavy_lift_end]
        df_vessel_util_feeder = df_vessel_util.loc[feeder_start:feeder_end]
        df_vessel_util_ahts = df_vessel_util.loc[ahts_start:ahts_end]

        df_vessel_count_wtiv = df_vessel_count.loc[wtiv_start:wtiv_end]
        df_vessel_count_wtiv_us = df_vessel_count.loc[wtiv_us_start:wtiv_us_end]
        df_vessel_count_heavy_lift = df_vessel_count.loc[heavy_lift_start:heavy_lift_end]
        df_vessel_count_feeder = df_vessel_count.loc[feeder_start:feeder_end]
        df_vessel_count_ahts = df_vessel_count.loc[ahts_start:ahts_end]

        df_perc_util_wtiv = df_vessel_util_wtiv / df_vessel_count_wtiv / 8766 * 100
        df_perc_util_wtiv_us = df_vessel_util_wtiv_us / df_vessel_count_wtiv_us / 8766 * 100
        df_perc_util_heavy_lift = df_vessel_util_heavy_lift / df_vessel_count_heavy_lift / 8766 * 100
        df_perc_util_feeder = df_vessel_util_feeder / df_vessel_count_feeder / 8766 * 100
        df_perc_util_ahts = df_vessel_util_ahts / df_vessel_count_ahts / 8766 * 100

        avg_utilization['example_wtiv']._append(df_perc_util_wtiv['example_wtiv'].mean())
        avg_utilization['example_wtiv_us']._append(df_perc_util_wtiv_us['example_wtiv_us'].mean())
        avg_utilization['example_heavy_lift_vessel']._append(df_perc_util_heavy_lift['example_heavy_lift_vessel'].mean())
        avg_utilization['example_feeder']._append(df_perc_util_feeder['example_feeder'].mean())
        avg_utilization['example_ahts_vessel']._append(df_perc_util_ahts['example_ahts_vessel'].mean())

    df_avg_utilization = pd.DataFrame(avg_utilization, index=desc)

    output_path = 'analysis/results/Select_Optimal_Scenarios'
    os.makedirs(output_path, exist_ok=True)
    csv_file_path = os.path.join(output_path, 'average_vessel_utilization.csv')
    df_avg_utilization.to_csv(csv_file_path)

    return avg_utilization

def run_plots(prs, log, history, ports, summary_table_filename):
    ne = ['MA','ME','CT','RI','NH','RI/CT']
    nynj = ['NY','NJ']
    mid = ['NC', 'MD', 'VA', 'DE']

    history = plot_shared_resource_capacities(prs, history)
    summary_table = percent_resource_demand(history, summary_table_filename)


    full_gantt(prs, log)
    full_gantt(prs, log, sorted=True)

    # regional_gantt(prs, log, ne, 'New England')
    # regional_gantt(prs, log, ne, 'New England', sorted=True)

    # port_gantts(prs, log, ports)
    # port_gantts(prs, log, ports, sorted=True)

    # substructure_gantt(prs, log, 'fixed')
    # substructure_gantt(prs, log, 'fixed', sorted=True)
    # substructure_gantt(prs, log, 'floating')
    # substructure_gantt(prs, log, 'floating', sorted=True)

    # vessel_utilization_plot(prs,log)

    # port_throughput(prs,log)
    # port_throughput(prs,log,ne)
    # port_throughput(prs,log,nynj)
    # port_throughput(prs,log,mid)

## Summary Plots ##
   
def installed_cap(prs, logs, desc, region = None):
    yrs = np.arange(2023,2043,1)
    df_cap = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)
    df_cum = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)

    log = logs[0]
    if region:
        log = log.drop(columns=['index'])
        log = log[log['location'].isin(region)].reset_index(drop=True).reset_index()

    log['cod'] = log['estimated_cod'].dt.year
    df_cod = log.groupby(['cod']).capacity.sum().reset_index()
    df_cod['sum'] = df_cod['capacity'].cumsum(axis=0) / 1000
    # print(df_cod)
    # df_cum['cod'] = df_cod['sum']
    
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(1,1,1)
    df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

    i=0
    width = 0.25

    for log in logs:
        log['finished'] = log['Date Finished'].dt.year
        if region:
            log = log.drop(columns=['index'])
            log = log[log['location'].isin(region)].reset_index(drop=True).reset_index()
        df_finished = log.groupby(['finished']).capacity.sum().reset_index()
        df_finished['capacity'] = df_finished['capacity'] / 1000
        df_finished['sum'] = df_finished['capacity'].cumsum(axis=0)

        cap_mapping = dict(df_finished[['finished', 'capacity']].values)
        df_cap[desc[i]] = df_cap.index.map(cap_mapping).fillna(0)

        df_cum[desc[i]] = df_cap[desc[i]].cumsum(axis=0)
        i += 1
    

    # colors = {'natl_gaps_2us': 'tab:orange','natl_gaps_3foreign':'tab:blue','natl_gaps_6AHTS':'tab:red', 'natl_gaps_no_action': 'tab:purple'}
    df_cum[desc].plot(linestyle = '-', ax=ax, label='cumulative')
    # df_cap[desc].plot(kind='bar', ax=ax, label='annual')
    ax.set_xlabel("")
    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xticks(np.arange(2022,2043,5))
    cum_label = [s + ' cumulative' for s in desc]
    labels = ['cod'] + cum_label
    #ax.legend(labels)
    ax.legend(labels, prop={'size': 7})

    slide = add_to_pptx(prs,'Cumulative Installed Capacity')

    return df_cum

def compare_installed_cap(prs, logs, desc, region=None):

    df_2040 = pd.DataFrame(columns = ['2040'])
    df_2030 = pd.DataFrame(columns = ['2040'])
    df_2050 = pd.DataFrame(columns = ['2050'])
    
    i=0
    for log in logs:
        if region:
            log = log.drop(columns=['index'])
            log = log[log['location'].isin(region)].reset_index(drop=True).reset_index()
        cap_by_year = pd.DataFrame()
        cap_by_year['year'] = pd.DatetimeIndex(log['Date Finished']).year
        cap_by_year['capacity'] = log['capacity']
        cap = cap_by_year.groupby(['year'])['capacity'].sum().reset_index()
        cap_2030 = cap.loc[cap['year'] <= 2030]['capacity'].sum()/1e3
        cap_2040 = cap.loc[cap['year'] <= 2040]['capacity'].sum()/1e3
        cap_2050 = cap.loc[cap['year'] <= 2050]['capacity'].sum()/1e3
        row_2030 = {'Scenario': desc[i], '2030': cap_2030}
        row_2040 = {'Scenario': desc[i], '2040': cap_2040}
        row_2050 = {'Scenario': desc[i], '2050': cap_2050}
        df_2030 = df_2030._append(row_2030, ignore_index=True)
        df_2040 = df_2040._append(row_2040, ignore_index=True)
        df_2050 = df_2050._append(row_2050, ignore_index=True)
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

    # fig = plt.figure(figsize=(6,4), dpi=200)
    # ax = fig.add_subplot(111)

    # df_caps.plot.bar(rot=0, ax=ax, width=0.3)

    # ax.set_ylabel('Installed Capacity (GW)')
    # ax.set_xlabel('')
    # ax.set_xlim(-0.25,3.25)

    # for p in ax.patches:
    #     ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height() * 1.005), fontsize=6)

    # colors = {'2030 Capacity':'tab:blue', 
    #           '2040 Capacity':'tab:orange',
    #           '2040 Capacity per # WTIV':'tab:green'}

    # labels = list(colors.keys())
    # handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    # ax.legend(handles, labels, loc='upper left', prop={'size': 6})
    # slide = add_to_pptx(prs,'Summary Installed Cap')

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
    slide = add_to_pptx(prs,'Summary Installed Capacity')

    plt.close()





