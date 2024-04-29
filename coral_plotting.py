from coral_imports import *


def add_text_slide(prs, title, text, left=0, top=7.2, width=13.33, height=0.3, fontsize=14):
    """Add text slide for scenarion description"""
    blank_slide_layout = prs.slide_layouts[1]
    slide = prs.slides.add_slide(blank_slide_layout)
    slide.shapes.title.text = title

    text_shape = slide.shapes.placeholders[10]

    text_frame = text_shape.text_frame

   
    text_frame.text = text[0]

    for para_str in text[1:]:
        p = text_frame.add_paragraph()
        p.text = para_str
        # p.level = 1




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

    fig = plt.figure(figsize=(8, len(df)/4), dpi=200)
    ax = fig.add_subplot(111)

    bar_color = []
    for i,row in df.iterrows():
        if row['substructure'] == 'fixed':
            bar_color.append("#FFD700")
        else:
            bar_color.append("#069AF3")

    df["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Installation Time", color = bar_color)
    df["Date Started"].plot(kind="barh", ax=ax, zorder=4, label="Delay", color="#E34234")
    df["Date Initialized"].plot(kind="barh", color="w", ax=ax, zorder=4, label="__nolabel__")

    df.plot(kind="scatter", x="Date Initialized", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker="d")
    
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df['name'])

    ax.legend()

    ax.set_xlim(manager._start - dt.timedelta(days=30), df["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted Full Gantt')
    else:
        slide = add_to_pptx(prs,'Full Gantt')
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
    for i,row in df.iterrows():
        if row['substructure'] == 'fixed':
            bar_color.append("#FFD700")
        else:
            bar_color.append("#069AF3")

    df_region["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
    df_region["Date Started"].plot(kind="barh", color="#E34234", ax=ax, zorder=4, label="Delay")
    df_region["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label = "__nolabel__", color = 'w')

    df_region.plot(kind="scatter", x="Date Initialized", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker="d")

    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df_region['name'])

    ax.legend()

    ax.set_xlim(manager._start - dt.timedelta(days=30), df_region["Date Finished"].max() + dt.timedelta(days=30))
    if sorted:
        slide = add_to_pptx(prs,'Sorted %s Gantt' % region_name)
    else:
        slide = add_to_pptx(prs,'%s Gantt' % region_name)
    plt.close(fig)


def substructure_gantt(prs, manager, df, substructure, sorted=False):
    """ Gantt filtered by either fixed or floating projects. Sorted sorts by expected start date."""

    df = df.drop(columns=['index'])
    df = df[df['substructure'] == substructure].reset_index(drop=True).reset_index()
    if sorted:
        df = df.drop(columns=['index'])
        df = df.sort_values(by=['Date Initialized'], ascending=False).reset_index(drop=True).reset_index()

    fig = plt.figure(figsize=(8, len(df)/4), dpi=200)
    ax = fig.add_subplot(111)

    df["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Installation Time", color="#B5B5B5")
    df["Date Started"].plot(kind="barh", ax=ax, zorder=4, label="Delay", color="#E34234")
    df["Date Initialized"].plot(kind="barh", color="w", ax=ax, zorder=4, label="__nolabel__")

    df.plot(kind="scatter", x="Date Initialized", y="index", color='k', ax=ax, zorder=5, label="Expected Start", marker="d")

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
        for _,row in df_port.iterrows():
            if row['substructure'] == 'fixed':
                bar_color.append("#FFD700")
            else:
                bar_color.append("#069AF3")

        df_port["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=bar_color)
        df_port["Date Started"].plot(kind="barh", color="#E34234", ax=ax, zorder=4, label="Delay")
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


def installed_cap(prs, dfs, desc):
    dfs[0]['cod'] = dfs[0]['estimated_cod'].dt.year
    df_cod = dfs[0].groupby(['cod']).capacity.sum().reset_index()
    df_cod['sum'] = df_cod['capacity'].cumsum(axis=0) / 1000

    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = fig.add_subplot(1,1,1)

    df_cod.plot(kind='line', x='cod', y='sum', color='k', ax=ax, label='Unconstrained Resources')
    i=0
    for df in dfs:
        df['finished'] = df['Date Finished'].dt.year
        df_finished = df.groupby(['finished']).capacity.sum().reset_index()
        df_finished['sum'] = df_finished['capacity'].cumsum(axis=0) / 1000
        df_finished.plot(kind='line', x='finished', y='sum', ax=ax, label=desc[i])
        i += 1
    ax.set_xlabel("")
    ax.set_ylabel("Capacity (GW)")
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend()

    slide = add_to_pptx(prs,'Cummulative Installed Capacity')

def run_plots(prs, manager, df):
    ne = ['MA','ME','CT','RI','NH','RI/CT']
    nynj = ['NY','NJ']
    mid = ['NC', 'MD', 'VA', 'DE']

    ports = ['searsport', 'new_bedford', 'new_london', 'njwp', 'sbmt', 'tradepoint', 'portsmouth']
    ne_ports = ['searsport','new_bedford','new_london']
    full_gantt(prs, manager, df)
    full_gantt(prs, manager, df, sorted=True)

    regional_gantt(prs, manager, df, ne, 'New England')
    regional_gantt(prs, manager, df, ne, 'New England', sorted=True)

    regional_gantt(prs, manager, df, nynj, 'New York/New Jersey')
    regional_gantt(prs, manager, df, nynj, 'New York/New Jersey', sorted=True)

    regional_gantt(prs, manager, df, mid, 'Midatlantic')
    regional_gantt(prs, manager, df, mid, 'Midatlantic', sorted=True)

    port_gantts(prs, manager, df, ports)
    port_gantts(prs, manager, df, ports, sorted=True)

    substructure_gantt(prs, manager, df, 'fixed')
    substructure_gantt(prs, manager, df, 'fixed', sorted=True)
    substructure_gantt(prs, manager, df, 'floating')
    substructure_gantt(prs, manager, df, 'floating', sorted=True)