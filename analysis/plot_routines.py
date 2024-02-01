import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as txt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import os
import datetime as dt
import pandas as pd
import numpy as np

from CORAL.utils import get_installed_capacity_by


def mysave(fig, froot, mode='png'):
    assert mode in ['png', 'eps', 'pdf', 'all']
    fileName, fileExtension = os.path.splitext(froot)
    padding = 0.1
    dpiVal = 200
    legs = []
    for a in fig.get_axes():
        addLeg = a.get_legend()
        if not addLeg is None: legs.append(a.get_legend())
    ext = []
    if mode == 'png' or mode == 'all':
        ext.append('png')
    if mode == 'eps':  # or mode == 'all':
        ext.append('eps')
    if mode == 'pdf' or mode == 'all':
        ext.append('pdf')

    for sfx in ext:
        fig.savefig(fileName + '.' + sfx, format=sfx, pad_inches=padding, bbox_inches='tight',
                    dpi=dpiVal, bbox_extra_artists=legs)


titleSize = 40  # 24 #38
axLabelSize = 38  # 20 #36
tickLabelSize = 40  # 18 #28
ganttTick = 18
legendSize = tickLabelSize + 2
textSize = legendSize - 2
deltaShow = 4
linewidth = 3


def myformat(ax, linewidth=linewidth, xticklabel=tickLabelSize, yticklabel=tickLabelSize, mode='save'):
    assert type(mode) == type('')
    assert mode.lower() in ['save', 'show'], 'Unknown mode'

    def myformat(myax, linewidth=linewidth, xticklabel=xticklabel, yticklabel=yticklabel):
        if mode.lower() == 'show':
            for i in myax.get_children():  # Gets EVERYTHING!
                if isinstance(i, txt.Text):
                    i.set_size(textSize + 3 * deltaShow)

            for i in myax.get_lines():
                if i.get_marker() == 'D': continue  # Don't modify baseline diamond
                i.set_linewidth(linewidth)
                # i.set_markeredgewidth(4)
                i.set_markersize(10)

            leg = myax.get_legend()
            if not leg is None:
                for t in leg.get_texts(): t.set_fontsize(legendSize + deltaShow + 6)
                th = leg.get_title()
                if not th is None:
                    th.set_fontsize(legendSize + deltaShow + 6)

            myax.set_title(myax.get_title(), size=titleSize + deltaShow, weight='bold', pad=20)
            myax.set_xlabel(myax.get_xlabel(), size=axLabelSize + deltaShow, weight='bold')
            myax.set_ylabel(myax.get_ylabel(), size=axLabelSize + deltaShow, weight='bold')
            myax.tick_params(labelsize=tickLabelSize + deltaShow)
            myax.patch.set_linewidth(3)
            for i in myax.get_xticklabels():
                i.set_size(tickLabelSize + deltaShow)
            for i in myax.get_xticklines():
                i.set_linewidth(3)
            for i in myax.get_yticklabels():
                i.set_size(yticklabel + deltaShow)
            for i in myax.get_yticklines():
                i.set_linewidth(3)

        elif mode.lower() == 'save':
            for i in myax.get_children():  # Gets EVERYTHING!
                if isinstance(i, txt.Text):
                    i.set_size(textSize)

            for i in myax.get_lines():
                if i.get_marker() == 'D': continue  # Don't modify baseline diamond
                i.set_linewidth(linewidth)
                # i.set_markeredgewidth(4)
                i.set_markersize(10)

            leg = myax.get_legend()
            if not leg is None:
                for t in leg.get_texts(): t.set_fontsize(legendSize)
                th = leg.get_title()
                if not th is None:
                    th.set_fontsize(legendSize)

            myax.set_title(myax.get_title(), size=titleSize, weight='bold', pad=20)
            myax.set_xlabel(myax.get_xlabel(), size=axLabelSize, weight='bold')
            myax.set_ylabel(myax.get_ylabel(), size=axLabelSize, weight='bold')
            myax.tick_params(labelsize=tickLabelSize)
            myax.patch.set_linewidth(3)
            for i in myax.get_xticklabels():
                i.set_size(xticklabel)
            for i in myax.get_xticklines():
                i.set_linewidth(3)
            for i in myax.get_yticklabels():
                i.set_size(yticklabel)
            for i in myax.get_yticklines():
                i.set_linewidth(3)

    if type(ax) == type([]):
        for i in ax: myformat(i)
    else:
        myformat(ax)

def initFigAxis(figx=32, figy=24):
    fig = plt.figure(figsize=(figx, figy))
    ax = fig.add_subplot(111)
    return fig, ax

def plot_gantt(df, manager, s, color_by, fname=None):
    fig, ax = initFigAxis()

    assign_colors(df, color_by)

    df = df.sort_values(by = ['region', 'Date Finished'], ascending=False)

    regions = df.region.unique()

    counts = []
    count = 0
    for r in regions:
        count += df['region'].value_counts()[r]
        counts.append((r, count))
    total = count

    df["y-labels"] = "    "
    order = range(1, total+1)
    df['order'] = order
    df.set_index('order', inplace=True)

    for group in counts:
        index = group[1]
        df.at[index - 2, 'y-labels'] = group[0]

    df["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=df["install color"])
    df["Date Started"].plot(kind="barh", color=df["delay color"], ax=ax, zorder=4, label="Project Delay", hatch="//", linewidth=0.5)
    df["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label="__nolabel__", color='w')

    port_base_handles = [
    Patch(facecolor=color, label=label)
    #for label, color in zip(['Humboldt', 'Coos Bay', 'Port San Luis', 'Long Beach', 'Grays Harbor'], ['#F39C12', '#16A085', '#C0392B', '#8E44AD', '#3498DB'])
    for label, color in zip(['Northern CA', 'Central OR', 'Central Coast (CA)', 'Southern CA', 'Southern WA'], ['#F39C12', '#16A085', '#C0392B', '#8E44AD', '#3498DB'])
    ]

    if "Southern WA" in df[color_by]:
        handles = region_exp_handles
    elif color_by == "port":
        handles = port_base_handles
    else:
        handles = region_base_handles

    # Plot formatting
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df['y-labels'])

    plt.yticks(fontsize=10)
    plt.plot((0, 0), (0, 30), scaley = False)

    ax.set_xlim(manager._start - dt.timedelta(days=30), dt.date(2060, 6, 1) + dt.timedelta(days=30))
    num_proj = len(df['Date Finished'])

    ax.axvline(dt.date(2046, 1, 1), lw=0.5, color="#2C3E50", zorder=6)
    installed_capacity_46 = get_installed_capacity_by(df, 2046)/1000

    for line in counts:
        ax.axhline(y = (line[1] - 0.5), ls="--", color="#979A9A")

    fig.subplots_adjust(left=0.25)

    plt.title(f"{s} scenario: \n{installed_capacity_46:,.3} GW of capacity installed by the end of 2045")

    # if s == '25 GW - High (SC)':
    ax.legend(handles=handles, loc = 'upper right', fontsize = 80, title="S&I Port")
    ax.text(x=dt.date(2046, 6, 1), y=(0.1*num_proj), s=f"End of 2045", fontsize=30, color="#2C3E50")

    if fname is not None:
        myformat(ax)
        mysave(fig, fname)
        plt.close()

## Bar chart of annual throughput of each port
def plot_throughput(throughput, fname=None):
    fig, ax = initFigAxis()

    throughput.plot.bar(ax=ax, width=0.75)

    ax.set_ylim(0, 2000)

    ax.set_ylabel("Annual Capacity Throughput (MW)")
    ax.set_xlabel("")

    plt.xticks(rotation=0, fontsize=6)
    plt.yticks(fontsize=6)

    ax.legend(fontsize=6, ncol=5)

    if fname is not None:
        myformat(ax)
        mysave(fig, fname)
        plt.close()
    #fname_t = 'results/throughput_'+str(s)+'.png'
    #fig.savefig(fname_t, dpi=300)

## Plot a near-term gantt chart
def plot_gantt_nt(df, manager, num_proj, color_by, fname=None):
    fig, ax = initFigAxis()

    assign_colors(df, color_by)

    df_nt = df.tail(num_proj)

    df_nt["Date Finished"].plot(kind="barh", ax=ax, zorder=4, label="Project Time", color=df_nt["install color"])
    df_nt["Date Started"].plot(kind="barh", color=df_nt["delay color"], ax=ax, zorder=4, label="Project Delay", hatch="////", linewidth=0.5)
    df_nt["Date Initialized"].plot(kind='barh', ax=ax, zorder=4, label="__nolabel__", color='w')

    # Plot formatting
    ax.set_xlabel("")
    ax.set_ylabel("")
    _ = ax.set_yticklabels(df_nt['region'])

    plt.yticks(fontsize=6)
    plt.plot((0, 0), (0, 30), scaley = False)
    ax.legend()
    ax.set_xlim(manager._start - dt.timedelta(days=30), dt.date(2040, 6, 1) + dt.timedelta(days=30))

    ax.axvline(dt.date(2031, 1, 1), lw=0.5, ls="--", color="#2C3E50", zorder=6)
    installed_capacity_31 = get_installed_capacity_by(df, 2031)
    ax.text(x=dt.date(2037, 1, 1), y=25, s=f"Capacity installed \nby end of 2030: \n{installed_capacity_31/1000:,.3} GW", fontsize=24, color="#2C3E50")

    fig.subplots_adjust(left=0.25)

    if fname is not None:
        myformat(ax)
        mysave(fig, fname)
        plt.close()

def assign_colors(df, color_by):

    delay_color = []
    install_color = []

    for index, row in df.iterrows():
        if df[color_by][index] == "Northern CA":
            delay_color.append("#F5B7B1")
            install_color.append("#E74C3C")
        elif df[color_by][index] == "Central CA":
            delay_color.append("#D2B4DE")
            install_color.append("#8E44AD")
        elif df[color_by][index] == "Central OR":
            delay_color.append("#AED6F1")
            install_color.append("#3498DB")
        elif df[color_by][index] == "Southern OR":
            delay_color.append("#F9E79F")
            install_color.append("#F1C40F")
        elif df[color_by][index] == "Southern WA":
            delay_color.append("#A9DFBF")
            install_color.append("#27AE60")
        elif df[color_by][index] == "Humboldt":
            delay_color.append("#FAD7A0")
            install_color.append("#F39C12")
        elif df[color_by][index] == "Coos Bay":
            delay_color.append("#A2D9CE")
            install_color.append("#16A085")
        elif df[color_by][index] == "Port of San Luis":
            delay_color.append("#E6B0AA")
            install_color.append("#C0392B")
        elif df[color_by][index] == "Long Beach":
            delay_color.append("#D2B4DE")
            install_color.append("#8E44AD")
        elif df[color_by][index] == "Grays Harbor":
            delay_color.append("#AED6F1")
            install_color.append("#3498DB")
        else:
            delay_color.append("#e9e9e9")
            install_color.append("#e9e9e9")

    df["delay color"] = delay_color
    df["install color"] = install_color

def plot_summary(scenarios, capacity_list, target_capacity):
    by_year = 2045

    inv_df = pd.read_excel('library/investments/scenario-investments.xlsx', sheet_name='schedule')
    inv_df = inv_df.set_index('Year')

    invest_list = []
    for s in scenarios:
        invest_list.append(inv_df[s][by_year])

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    width = 0.25

    x_ind = np.arange(len(scenarios))

    target = [int(s.split(' ')[0]) for s in scenarios]
    ax1.plot(x_ind-(width/2), target, width, color='#2ECC71', marker='*', markersize = 12, linestyle="", zorder=0)

    not_installed = []
    for goal, actual in zip(target, capacity_list):
        dif = goal-actual
        not_installed.append(dif)
    df_installs = pd.DataFrame(list(zip(scenarios, capacity_list, not_installed)), columns=['Scenarios', 'Installed', 'Not installed'])

    ax1.bar(x_ind-(width/2), capacity_list, width, color='#2874A6', zorder=5)

    ax1.set_xlabel('Installation scenario', weight='bold')
    ax1.set_ylabel('Installed capacity by end of ' + str(by_year) + ', GW', weight='bold')
    ax1.set_ylim([0,60])

    perc_installed = [round(100*c/t, 3) for c,t in zip(capacity_list, target)]
    perc_installed_dict = {}
    for s,p in zip(scenarios, perc_installed):
        perc_installed_dict[s] = p

    ax2.bar(x_ind+(width/2), invest_list, width, color='#F39C12')
    ax2.set_ylabel('Investment required, $ billion', weight='bold')
    ax2.set_ylim([0,11.5])
    ax2.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax1.set_xticks(x_ind)
    plot_names = ['25 GW (SC) \n(4 S&I sites)', '25 GW (CC) \n(3 S&I sites)', '35 GW \n(6 S&I sites)', '55 GW \n(9 S&I  sites)']

    #num = len(scenarios)
    if len(scenarios) > 1:
        ax1.set_xticklabels(plot_names, rotation=45)

    t_star = mlines.Line2D([], [], color='#2ECC71', marker='*', linestyle='None', markersize=12, label='Target capacity')
    installed_blue = mlines.Line2D([], [], color='#2874A6', marker='s', linestyle='None', markersize=10, label='Installed capacity')
    investment_orange = mlines.Line2D([], [], color='#F39C12', marker='s', linestyle='None', markersize=10, label='Investment')

    ax1.legend(handles=[t_star, installed_blue, investment_orange], loc='upper left');

    fig.savefig('results/Summary Plots/summary.png', bbox_inches='tight', dpi=300)

    return perc_installed_dict

def plot_deployment():
    levels = ['25 GW', '35 GW', '55 GW']

    schedules = 'library/pipeline/deployment-schedules.xlsx'

    for s in levels:
        df = pd.read_excel(schedules, sheet_name = s, index_col='Year')
        df = df/1000

        if s=='25 GW':
            regions = df[['Central CA', 'Northern CA']].copy()
            total = regions.sum(axis=1)[2045]
        elif s=='35 GW':
            regions = df[['Central CA', 'Northern CA', 'Central OR', 'Southern OR']].copy()
            total = regions.sum(axis=1)[2045]
        elif s =='55 GW':
            regions = df[['Central CA', 'Northern CA', 'Central OR', 'Southern OR', 'Southern WA']].copy()
            total = regions.sum(axis=1)[2045]

        fig, ax = initFigAxis()
        ax2 = ax.twinx()

        ## Stacked area charts
        ax = regions.plot.area(alpha=0.75)
        plt.title('Target Deployment for the '+ s +' Scenario' )
        ax.set_ylabel('Cumulative installed capacity, GW')
        ax.legend(loc = 'upper left')

        area_fname = 'results/Deployment/' + s + '_stacked.png'
        plt.savefig(area_fname, bbox_inches='tight')

        ## Simple line graphs
        ax2 = regions.plot.line()
        plt.title('Target Deployment for the '+ s +' Scenario' )
        ax2.set_ylim([0,25])
        ax2.set_ylabel('Cumulative installed capacity, GW')
        ax2.legend(loc = 'upper left')

        line_fname = 'results/Deployment/' + s + '_line.png'
        plt.savefig(line_fname, bbox_inches='tight')

        myformat(ax)
        myformat(ax2)
        #mysave(fig, line_fname)

def plot_investments(cap_dir, scenarios):
    fig, ax1 = initFigAxis()
    ax2 = ax1.twinx()

    inv_df = pd.read_excel('library/investments/scenario-investments.xlsx', sheet_name='schedule')

    for s in scenarios:
        ax1.plot(inv_df['Year'], inv_df[s])

        inst_df = pd.read_excel(cap_dir, sheet_name = s)
        ax2.plot(inst_df['Year'], inst_df['Cumulative Capacity'])

    inv_name = 'results/Summary Plots/investments.png'
    plt.savefig(inv_name, bbox_inches='tight')

def plot_per_dollar(scenarios, percent_installed, target_capacity):
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    width = 0.25

    x_ind = np.arange(len(scenarios))
    ax1_ind = np.arange(0, 11, 2)
    #ax2_ind = np.arange(0, 11, 1)
    width = 0.25
    
    # target = [target_capacity[s.split(' -')[0]] for s in scenarios]
    target = [int(s.split(' ')[0]) for s in scenarios]

    percents = []
    for s in percent_installed:
        percents.append(percent_installed[s])

    installed = []
    for t, p in zip(target, percents):
        actual_inst = t*p/100
        installed.append(actual_inst)

    inv_df = pd.read_excel('library/investments/scenario-investments.xlsx', sheet_name='schedule')
    inv_df = inv_df.set_index('Year')
    by_year = 2045
    invest_list = []
    for s in scenarios:
        invest_list.append(inv_df[s][by_year])

    per_dollar = []
    for c, d in zip(installed, invest_list):
        installed_per_dollar = c/d
        per_dollar.append(installed_per_dollar)

    ax1.bar(x_ind-width/2, per_dollar, width=width, color='#9B59B6')
    ax1.set_xlabel('GW of capacity installed per billion USD', weight='bold')
    ax1.set_ylabel('Installation scenario', weight='bold')
#    ax1.set_xticks(ax1_ind)
    plot_names = scenarios
    ax1.set_yticklabels(plot_names)
    ax1.set_ylim([0,11])
#    ax1.set_title('S&I port investment efficiency')

    ax2.bar(x_ind+width/2, percents, width=width, color='#F1C40F')
    ax2.set_ylabel('Percent of target installed per million dollars invested')
    ax2.set_ylim([0,100])

    handles = [
        Patch(facecolor=color, label=label)
#        for label, color in zip(['Percent of target', 'Investment efficiency'], ['#9B59B6', '#F1C40F'])
        for label, color in zip(['Percent of target'], ['#9B59B6'])
    ]

#    ax1.legend(handles=handles, loc='upper left')

    figsave = 'results/Summary Plots/per_dollar.png'
    fig.savefig(figsave, bbox_inches='tight', dpi=300)

def plot_new_gantt(df, manager, s, color_by, inv_df, fname=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False, gridspec_kw={'height_ratios': [3,1]})
    fig.subplots_adjust(hspace=0)

    assign_colors(df, color_by)

    df = df.sort_values(by = ['region', 'Date Finished'], ascending=False)

    regions = df.region.unique()

    counts = []
    count = 0
    for r in regions:
        count += df['region'].value_counts()[r]
        counts.append((r, count))
    total = count

    df["y-labels"] = "   "
    order = range(1, total+1)
    df['order'] = order
    df.set_index('order', inplace=True)

    for group in counts:
        index = group[1]
        df.at[index - 2, 'y-labels'] = group[0]

    df["Date Finished"].plot(kind="barh", ax=ax1, zorder=4, label="Project Time", color=df["install color"])
    df["Date Started"].plot(kind="barh", color=df["delay color"], ax=ax1, zorder=4, label="Project Delay", hatch="////", linewidth=0.5)
    df["Date Initialized"].plot(kind='barh', ax=ax1, zorder=4, label="__nolabel__", color='w')

    port_base_handles = [
    Patch(facecolor=color, label=label)
    #for label, color in zip(['Humboldt', 'Coos Bay', 'Port San Luis', 'Long Beach', 'Grays Harbor'], ['#F39C12', '#16A085', '#C0392B', '#8E44AD', '#3498DB'])
    for label, color in zip(['Northern CA', 'Central OR', 'CA Central Coast', 'Southern CA', 'Southern WA', 'Cumulative \ninvestment'], ['#F39C12', '#16A085', '#C0392B', '#8E44AD', '#3498DB', 'r'])
    ]

    if "Southern WA" in df[color_by]:
        handles = region_exp_handles
    elif color_by == "port":
        handles = port_base_handles
    else:
        handles = region_base_handles

    # Plot formatting
    ax1.set_xlabel(" ")
    ax1.set_ylabel("Region", weight='bold')
    _ = ax1.set_yticklabels(df['y-labels'])
    ax1.xaxis.set_tick_params(labelbottom=False)


    plt.yticks(fontsize=10)
    plt.plot((0, 0), (0, 30), scaley = False)

    #ax1.set_xlim(manager._start - dt.timedelta(days=30), dt.date(2060, 6, 1) + dt.timedelta(days=30))
    ax1.set_xlim(dt.date(2027, 1, 1) + dt.timedelta(days=30), dt.date(2061, 1, 1) + dt.timedelta(days=30))
    num_proj = len(df['Date Finished'])

    # ax1.axvline(dt.date(2046, 2, 1), lw=0.5, color="#2C3E50", zorder=6)
    installed_capacity_46 = get_installed_capacity_by(df, 2046)/1000

    for line in counts:
        ax1.axhline(y = (line[1] - 0.5), ls="--", color="#979A9A")

    fig.subplots_adjust(left=0.25)

    inv_df.set_index("Year", inplace=True)
    invested = inv_df.at[2045, s]

    ax1.set_title(f"{s} scenario: {invested:,.1f} billion USD \ninvested and {installed_capacity_46:,.3} GW installed by the end of 2045", weight='bold')

    # if s == '25 GW - High (SC)':
    ax1.legend(handles=handles, title="S&I Port", bbox_to_anchor=[1.05, 1], loc='upper left')
    # ax1.text(x=dt.date(2046, 6, 1), y=(0.1*num_proj), s=f"End of 2045", color="#2C3E50")

    inv_df[s].plot.line(ax=ax2, color='r')
    # ax2.axvline(x=2046, lw=0.5, color='#2C3E50')
    ax2.set_xlim(2027, 2061)
    ax2.set_ylim(0, 11)
    ax2.set_ylabel("Investment \n(million USD)", weight='bold', rotation=90)
    ax2.yaxis.set_label_coords(-.2, .5)


    if fname is not None:
        mysave(fig, fname)
        plt.close()

def plot_total_investments(file_name):
    df = pd.read_excel(file_name, sheet_name = 'total-investments')

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    width=0.4

    df.plot.bar(x = 'Scenario', y = ['S&I', 'MF', 'O&M'], color=['#2980B9', '#E67E22', '#F1C40F'], stacked = True, width=width, ax=ax, rot=45)

    ax.set_ylabel('Investment ($ million)', weight='bold')
    ax.set_xlabel('Scenario', weight='bold')
    ax.set_title('Total scenario investments by type', weight='bold')
    fig.savefig('results/Summary Plots/total-investments.png', bbox_inches='tight', dpi=300)

def plot_deployment2():
    levels = ['25 GW target deployment', '35 GW target deployment', '55 GW target deployment']

    schedules = 'library/pipeline/deployment-schedules.xlsx'

    b_df = pd.read_excel(schedules, sheet_name = '25 GW', index_col='Year')
    m_df = pd.read_excel(schedules, sheet_name = '35 GW', index_col='Year')
    e_df = pd.read_excel(schedules, sheet_name = '55 GW', index_col='Year')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True, figsize=(10,12))

    baseline = b_df[['Overall', 'Central CA', 'Northern CA']].copy()
    moderate = m_df[['Overall', 'Central CA', 'Northern CA', 'Central OR', 'Southern OR']].copy()
    expanded = e_df[['Overall', 'Central CA', 'Northern CA', 'Central OR', 'Southern OR', 'Southern WA']].copy()
    s_list = [baseline, moderate, expanded]

    b_colors = ['b', 'g', 'r']
    m_colors = ['blue', 'green', 'orange', 'yellow', 'black']
    e_colors = ['blue', 'green', 'orange', 'yellow', 'purple', 'black']
    all_colors = [b_colors, m_colors, e_colors]

    for n, df, t, c in zip(np.arange(1,4), s_list, levels, all_colors):
        ax = plt.subplot(3, 1, n)
        ax.plot(df/1000)
        ax.set_title(t, weight='bold')
        ax.set_ylabel('Cumulative installed \ncapacity, GW', weight='bold')
        column_names = list(df.columns.values)
        ax.legend(labels=column_names, loc='upper left', title='Offshore wind region')

    line_fname = 'results/Deployment/stacked_subplots.png'
    plt.savefig(line_fname, bbox_inches='tight')
