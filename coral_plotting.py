from coral_imports import *


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

def full_gantt(prs, manager, df, sorted=False):
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

    ax.set_xlim(manager._start - dt.timedelta(days=30), df["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted Full Gantt', width=5.25)
    else:
        slide = add_to_pptx(prs,'Full Gantt', width=4.25)
    plt.close(fig)

def regional_gantt(prs, manager, df, region, region_name, sorted=False):
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


    ax.set_xlim(manager._start - dt.timedelta(days=30), df_region["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % region_name)
    else:
        slide = add_to_pptx(prs,'%s Gantt' % region_name)
    plt.close(fig)


def substructure_gantt(prs, manager, df, substructure, sorted=False):
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
            if row['port'] in ["new_bedford", "sbmt", "tradepoint"]:
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

    ax.set_xlim(manager._start - dt.timedelta(days=30), df["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % substructure.capitalize())
    else:
        slide = add_to_pptx(prs,'%s Gantt' % substructure.capitalize())
    plt.close(fig)


def port_gantts(prs, manager, df, ports, sorted=False):
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

        ax.set_xlim(manager._start - dt.timedelta(days=30), df_port["Date Finished"].max() + dt.timedelta(days=30))

        i += 1

    if sorted:
        slide = add_to_pptx(prs,'Sorted Port Gantts')
    else:
        slide = add_to_pptx(prs,'Port Gantts')    

    plt.close(fig)


def port_throughput(prs, manager, df, region=None):
    if region:
        df = df.drop(columns=['index'])
        df = df[df['offtake_state'].isin(region)].reset_index(drop=True).reset_index()

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

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(111)

    throughput.plot.bar(ax=ax, width=0.75)

    ax.set_ylim(0, 2000)
    ax.set_ylabel("Annual Capacity Throughput (MW)")
    ax.set_xlabel("")
    plt.xticks(rotation=0, fontsize=6)
    plt.yticks(fontsize=6)

    ax.legend(fontsize=6, ncol=5)

    slide = add_to_pptx(prs,'Port Throughput')


def vessel_investment_plot(prs, allocs, futures, names, vessel_types, vessel_costs):
    yrs = np.arange(2023,2060)
    dates = pd.to_datetime(yrs, format='%Y')
    fig, axes = plt.subplots(2,1, figsize=(10,6), dpi=200, sharex=True)

    us_investments = pd.DataFrame(index=dates, columns=names, data=np.zeros((len(yrs), len(names))))
    total_investments = pd.DataFrame(index=dates, columns=names, data=np.zeros((len(yrs), len(names))))
    foreign_investments = pd.DataFrame(index=dates, columns=names, data=np.zeros((len(yrs), len(names))))
    for i in range(0,len(names)):
        init_alloc = [allocs[i]['wtiv'][1][1], allocs[i]['wtiv'][0][1], allocs[i]['feeder'][1][1], allocs[i]['feeder'][1][1], allocs[i]['ahts_vessel'][1]]
        vessel_investment = pd.DataFrame(columns=vessel_types, data = np.zeros((len(yrs), len(vessel_types))), index = dates)
        vessel_investment.iloc[0] = init_alloc

        for vessel in vessel_types:
            for vessel_type in futures[i]:
                if vessel_type[1] == vessel:
                    years = vessel_type[2]
                    for year in years:
                        vessel_investment.loc[[year],vessel] += 1
            # vessel_investment[vessel] = vessel_investment[vessel] * vessel_costs[vessel]  # Just looking at # vessels rn
        
        us_vessels = ['example_feeder', 'example_heavy_feeder_1kit', 'example_ahts_vessel']
        foreign_vessels = ['example_wtiv', 'example_heavy_lift_vessel']
        vessel_investment.loc[:,'us_total'] = vessel_investment[us_vessels].sum(axis=1)
        vessel_investment['us_total'] = vessel_investment['us_total'].cumsum() # / 1000 <- for $M to $B
        vessel_investment.loc[:,'foreign_total'] = vessel_investment[foreign_vessels].sum(axis=1)
        vessel_investment['foreign_total'] = vessel_investment['foreign_total'].cumsum() # / 1000 <- for $M to $B
        vessel_investment.loc[:,'total'] = vessel_investment[vessel_types].sum(axis=1)
        vessel_investment['total'] = vessel_investment['total'].cumsum() # / 1000 <- for $M to $B

        total_investments[names[i]] = vessel_investment['total']
        us_investments[names[i]] = vessel_investment['us_total']
        foreign_investments[names[i]] = vessel_investment['foreign_total']
    
    total_investments['year'] = yrs
    total_investments.set_index('year', inplace=True)
    
    us_investments['year'] = yrs
    us_investments.set_index('year', inplace=True)

    us_investments['year'] = yrs
    us_investments.set_index('year', inplace=True)
    us_investments.plot(ax=axes[0])

    foreign_investments['year'] = yrs
    foreign_investments.set_index('year', inplace=True)
    foreign_investments.plot(ax=axes[1])
    axes[0].yaxis.set_major_locator(tck.MaxNLocator(integer=True))
    axes[0].set_ylabel('US (# vessels)')
    axes[1].set_ylabel('Foreign (# vessels)')
    plt.minorticks_off()
    axes[0].set_xticks(yrs[::2])

    slide = add_to_pptx(prs,'Vessel Investment')

    return total_investments

def installed_cap(prs, dfs, desc, region = None):
    # end = 2025
    # for df in dfs:
    #     print(df['Date Finished'].iloc[0].year)
    #     if df['Date Finished'].iloc[0].year > end:
    #         end = df['Date Finished'].iloc[-1].year
    yrs = np.arange(2023,2050,1)
    end = 2100
    for df in dfs:
        print(df['Date Finished'].iloc[0].year)
        if df['Date Finished'].iloc[0].year > end:
            end = df['Date Finished'].iloc[-1].year
    yrs = np.arange(2023,end,1)
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
    # df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax)

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
    
    df_cum[desc].plot(linestyle = '-', ax=ax, use_index=False, label='cumulative')
    df_cap[desc].plot(kind='bar', ax=ax, label='annual')
    ax.set_xlabel("")
    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    annual_label = [s + ' annual' for s in desc]
    cum_label = [s + ' cumulative' for s in desc]
    labels = cum_label + annual_label
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


def run_plots(prs, manager, df, ports):
    ne = ['MA','ME','CT','RI','NH','RI/CT']
    nynj = ['NY','NJ']
    mid = ['NC', 'MD', 'VA', 'DE']

    full_gantt(prs, manager, df)
    full_gantt(prs, manager, df, sorted=True)

    # regional_gantt(prs, manager, df, ne, 'New England')
    # regional_gantt(prs, manager, df, ne, 'New England', sorted=True)

    port_throughput(prs,manager,df)
    # port_throughput(prs,manager,df,ne)

    # regional_gantt(prs, manager, df, nynj, 'New York/New Jersey')
    # regional_gantt(prs, manager, df, nynj, 'New York/New Jersey', sorted=True)

    # regional_gantt(prs, manager, df, mid, 'Midatlantic')
    # regional_gantt(prs, manager, df, mid, 'Midatlantic', sorted=True)

    # port_gantts(prs, manager, df, ports)
    # port_gantts(prs, manager, df, ports, sorted=True)

    substructure_gantt(prs, manager, df, 'fixed')
    substructure_gantt(prs, manager, df, 'fixed', sorted=True)
    # substructure_gantt(prs, manager, df, 'floating')
    # substructure_gantt(prs, manager, df, 'floating', sorted=True)