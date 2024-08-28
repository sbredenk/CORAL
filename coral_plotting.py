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
    """Gantt chcart of select region pipeline. Region determined by offtake states in region list. 
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


def vessel_utilization_plot(prs, df):

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
    ax.legend(fontsize=6)
    ax.set_ylim(0,100)

    # slide = add_to_pptx(prs,'Vessel Utilization')
    return(df_vessel_util/24)


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


def vessel_investment_plot(prs, names):
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

    us_investments = pd.DataFrame(index=dates, columns=names, data=np.zeros((len(yrs), len(names))))
    total_investments = pd.DataFrame(index=dates, columns=names, data=np.zeros((len(yrs), len(names))))
    vessel_counts = []
    for i in range(0,len(names)):
        scen = read_yaml(names[i], scen_path)
        alloc = scen['allocations']
        future = scen['future_resources']
        init_alloc = [alloc['wtiv'][1][1], 
                      alloc['wtiv'][2][1], 
                      alloc['wtiv'][0][1], 
                      alloc['ahts_vessel'][1], 
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
        total_investments[names[i]] = vessel_investment['total']
        us_investments[names[i]] = vessel_investment['us_total']

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
        

def summary_invest_plot(prs, us_investments, us_rev):
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


    us_investments.plot(ax=ax, label='US Investments', color = 'blue', style=linestyle)
    us_rev.plot(ax=ax,label='US cost to Developers', color = 'green', style=linestyle)

    ax.set_ylim(0,5)
    ax.set_ylabel('$ Billion')
    ax.set_xlabel('')
    ax.set_xlim(2025,2045)
    ax.set_xticks(np.arange(2025,2050,5))
    h, l = ax.get_legend_handles_labels()
    h.reverse()
    l = ['No Action','AHTS Emphasis','US WTIV Emphasis', 'US Feeder Emphasis']
    ph = [plt.plot([],marker="", ls="")[0]]*2
    handles = ph[:1] + h[4:] + ph[1:] + h[:4]
    labels = ["US Investment"] + l + ["US Cost to Developer"] + l
    leg = ax.legend(handles, labels, ncol = 2, loc='upper left', fontsize=8)  
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)  
    fig.tight_layout()
    slide = add_to_pptx(prs, 'US Investment Summary')




# def compare_investments(prs, df_shares, desc):
#     fig = plt.figure(figsize=(10,4), dpi=200)
#     ax = fig.add_subplot(111)

#     yrs = np.arange(2023,2065,1)
#     df_cum_investment = pd.DataFrame(columns=desc, data = np.zeros((len(yrs), len(desc))), index = yrs)
#     i=0
#     for df in df_investments:
#         df.loc[:,'total'] = df.sum(axis=1)
#         df_cum_investment[desc[i]] = df['total']
#         i+=1

#     df_cum_investment.plot(kind='line',ax=ax)
#     ax.set_ylabel("Annual Invesment ($M)")
#     ax.set_xlabel("")

#     slide = add_to_pptx(prs, 'Vessel Investment by Scenario')

#     return(df_cum_investment)
    


def installed_cap(prs, dfs, desc, region = None):
    yrs = np.arange(2023,2043,1)
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
    df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

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

    slide = add_to_pptx(prs,'Installed Capacity')

    return df_cum


def cap_per_investment(prs, df_cum, df_investments):
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(1,1,1)

    df = df_cum / df_investments
    
    df.plot(ax=ax)
    ax.set_ylabel('Capacity per Investment (MW/$)')
    slide = add_to_pptx(prs, 'Capacity per Investment (MW/$)')


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

