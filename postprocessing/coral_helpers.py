__author__ = "Sophie Bredenkamp"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Sophie Bredenkamp"
__email__ = "sophie.bredenkamp@nrel.gov"

from coral_imports import *

def run_manager(pipeline, allocations, library, weather=None, future_resources=None, future_remove=None):
    """
    Runs GlobalManager and returns logs and resource history.

    Parameters
    ----------
    pipeline : str
        Filepath for project pipeline.
    allocations : dict
        Number of each library item that exists in the shared environment.
    library : str
        Path to shared library items.
    weather : str (optional)
        Path to weather csv.
    future_resources : list (optional)
        Date and type of each resource to be added during run.
    future_remove : list (optional)
        Date and type of each resource to be removed during run.    
    """
    manager = GlobalManager(pipeline.configs, allocations, weather, library_path=library)

    if future_resources != None: 
        for i in future_resources:
            manager.add_future_resources(i[0], i[1], i[2])

    if future_remove != None: 
        for i in future_remove:
            manager.remove_future_resources(i[0], i[1], i[2])
                
    manager.run()

    # Format DataFrame for figure building
    log = pd.DataFrame(manager.logs).iloc[::-1]
    log = log.reset_index(drop=True).reset_index()

    df_cols = ['substructure','depth', 'location','foundation_port', 'turbine_port', 'capacity','us_wtiv']

    for col in df_cols:
        map = pipeline.projects[["name", col]].set_index("name").to_dict()[col]
        log[col] = [map[name] for name in log['name']]
    
    cod_map = pipeline.projects[["name", "estimated_cod"]].set_index("name").to_dict()['estimated_cod']
    log['estimated_cod'] = [cod_map[name] for name in log['name']]
    log['estimated_cod'] = pd.to_datetime(log['estimated_cod'], format='%Y')

    history = manager.resource_history
    history_df = pd.DataFrame(history)
    return manager, log, history_df


def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)


def read_yaml(scenario, path):
    """
    Read yaml file.

    Parameters
    ----------
    scenario : str
        Scenario name.
    path : str
        Filepath to scenario.
    """
    # Register the constructor with PyYAML
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    yaml_path = os.path.join(os.getcwd(), "%s/%s.yaml" % (path,scenario))
    with open(yaml_path) as f:
        scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return(scenario)


def vessel_hours(log):
    """
    Calculates hours of vessel utilization in given year.

    Parameters
    ----------
    log : DataFrame
       Log output of CORAL run 
    """
    yrs = np.arange(2023,2055)
    df_util = pd.DataFrame(columns = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder'], index=yrs)
    df_util = df_util.fillna(0)
    log['Date TurbineStart'] = pd.to_datetime(log['Date TurbineStart'])

    for _,project in log.iterrows():
        # FOUNDATIONS
        if project['substructure'] in ('monopile','jacket'):
            if project['Date FoundationFinished'].year == project['Date Started'].year:
                util = (project['Date FoundationFinished'].date() - project['Date Started'].date()).days * 24
                df_util.loc[project['Date FoundationFinished'].year,'example_heavy_lift_vessel'] += util
            else:
                total = project['Date FoundationFinished'].date() - project['Date Started'].date()
                for year in np.arange(project['Date Started'].year,project['Date FoundationFinished'].year + 1):
                    if year == project['Date Started'].year:
                        util = (dt.date(year + 1, 1, 1) - project["Date Started"].date()).days * 24
                    elif year == project['Date FoundationFinished'].year:
                        util = (project['Date FoundationFinished'].date() - dt.date(year,1,1)).days * 24
                    else:
                        util = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days * 24
                    df_util.loc[year,'example_heavy_lift_vessel'] += util

        # TURBINES
        if project['substructure'] in ('monopile','jacket'):
            if project['Date Finished'].year == project['Date TurbineStart'].year:
                util = (project['Date Finished'].date() - project['Date TurbineStart'].date()).days * 24
                if project['us_wtiv']:
                    df_util.loc[project['Date Finished'].year,'example_wtiv_us'] += util
                else:
                    df_util.loc[project['Date Finished'].year,'example_wtiv'] += util
                    df_util.loc[project['Date Finished'].year,'example_feeder'] += util * 2
            else:
                total = project['Date Finished'].date() - project['Date TurbineStart'].date()
                for year in np.arange(project['Date TurbineStart'].year,project['Date Finished'].year + 1):
                    if year == project['Date TurbineStart'].year:
                        util = (dt.date(year + 1, 1, 1) - project["Date TurbineStart"].date()).days * 24
                    elif year == project['Date Finished'].year:
                        util = (project['Date Finished'].date() - dt.date(year,1,1)).days * 24
                    else:
                        util = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days * 24

                    if project['us_wtiv']:
                        df_util.loc[year,'example_wtiv_us'] += util
                    else:
                        df_util.loc[year,'example_wtiv'] += util
                        df_util.loc[year,'example_feeder'] += util * 2

        else:
            if project['Date Finished'].year == project['Date Started'].year:
                util = (project['Date Finished'].date() - project['Date Started'].date()).days * 24
                df_util.loc[project['Date Finished'].year,'example_ahts_vessel'] += util
            else:
                total = project['Date Finished'].date() - project['Date Started'].date()
                for year in np.arange(project['Date Started'].year,project['Date Finished'].year + 1):
                    if year == project['Date Started'].year:
                        util = (dt.date(year + 1, 1, 1) - project["Date Started"].date()).days * 24
                    elif year == project['Date Finished'].year:
                        util = (project['Date Finished'].date() - dt.date(year,1,1)).days * 24
                    else:
                        util = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days * 24
                    df_util.loc[year,'example_ahts_vessel'] += util
        
    return(df_util)

def vessel_pipeline(allocations, futures):
    """
    Counts total vessel of each type in shared resources in each year.

    Parameters
    ----------
    allocations : dict
        Number of each library item that exists in the shared environment.
    futures : list
        List of vessel type and year for all added vessel resources.
    """
    yrs = np.arange(2023,2055)
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)
    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    init_alloc = [allocations['wtiv'][1][1], allocations['wtiv'][2][1], allocations['wtiv'][0][1], allocations['ahts_vessel'][1], allocations['feeder'][1][1]]
    vessel_count = pd.DataFrame(columns=vessel_types, data = np.ones((len(yrs), len(vessel_types))), index = yrs)
    vessel_count = vessel_count.mul(init_alloc)

    for vessel in vessel_types:
        for vessel_type in futures:
            if vessel_type[1] == vessel:
                years = [x.year for x in vessel_type[2]]
                for year in years:
                    vessel_count.loc[year:,vessel] += 1

    return(vessel_count)


def squarify(data):
    """
    Formats data in DataFrame.

    Parameters
    ----------
    data : list
        Resource history data list.
    """
    out = []
    for i, (time, cap) in enumerate(data):

        out.append((time, cap))
        try:
            if cap != data[i + 1][1]:
                out.append((data[i + 1][0], cap))

        except IndexError:
            pass

    return pd.DataFrame(out, columns=["time", "capacity"])