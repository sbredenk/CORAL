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
    df_vessel_count = vessel_pipeline(allocs,futures)
    df_perc_util = df_vessel_util / df_vessel_count / 8766 * 100

    df_perc_util.plot(kind='bar', ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Vessel Utilization (%)")
    ax.legend(fontsize=6)

    slide = add_to_pptx(prs,'Vessel Utilization')
    return(df_vessel_util)

def vessel_investment_plot(prs, df_vessel_util):

    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)
    
    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    vessel_rates = []
    rate_path = 'analysis/library/vessels'
    for vessel in vessel_types:
        vessel_yaml = read_yaml(vessel, rate_path)
        vessel_rate = vessel_yaml['vessel_specs']['day_rate'] / 24
        vessel_rates.append(vessel_rate)

    df_investment = df_vessel_util.mul(vessel_rates) / 1e6

    df_investment.plot(kind='bar', ax=ax)
    ax.set_ylabel("Annual Invesment ($M)")
    ax.set_xlabel("")

    slide = add_to_pptx(prs, 'Vessel Investment')
    return(df_investment)


def compare_investments(prs, df_investments, desc):
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)

    yrs = np.arange(2023,2065,1)
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
    
    df_cum[desc].plot(linestyle = '-', ax=ax, label='cumulative')
    # df_cap[desc].plot(kind='bar', ax=ax, label='annual')
    ax.set_xlabel("")
    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
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

