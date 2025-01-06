from coral_imports import *

# set up yaml reading
def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# run manager function 
def run_manager(pipeline, allocations, library, weather=None, future_resources=None, future_remove=None, sorted=False):
    manager = GlobalManager(pipeline.configs, allocations, weather, library_path=library)

    if future_resources != None: 
        for i in future_resources:
            manager.add_future_resources(i[0], i[1], i[2])

    if future_remove != None: 
        for i in future_remove:
            manager.remove_future_resources(i[0], i[1], i[2])
                
    manager.run()

    # Format DataFrame for figure building
    df = pd.DataFrame(manager.logs).iloc[::-1]
    df = df.reset_index(drop=True).reset_index()

    df_cols = ['substructure','depth', 'location','associated_port', 'capacity','us_wtiv']

    for col in df_cols:
        map = pipeline.projects[["name", col]].set_index("name").to_dict()[col]
        df[col] = [map[name] for name in df['name']]
    
    cod_map = pipeline.projects[["name", "estimated_cod"]].set_index("name").to_dict()['estimated_cod']
    df['estimated_cod'] = [cod_map[name] for name in df['name']]
    df['estimated_cod'] = pd.to_datetime(df['estimated_cod'], format='%Y')

    return manager, df


def read_yaml(scenario, path):
    # Register the constructor with PyYAML
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)
    yaml_path = os.path.join(os.getcwd(), "%s/%s.yaml" % (path,scenario))
    with open(yaml_path) as f:
        scenario = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return(scenario)


def vessel_hours(df):
    yrs = np.arange(2023,2055)
    df_util = pd.DataFrame(columns = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder'], index=yrs)
    df_util = df_util.fillna(0)
    df['Date TurbineStart'] = pd.to_datetime(df['Date TurbineStart'])

    for _,project in df.iterrows():
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

def vessel_pipeline(allocs, futures):
    yrs = np.arange(2023,2055)
    # dates = pd.to_datetime(yrs, format='%Y')
    fig = plt.figure(figsize=(10,4), dpi=200)
    ax = fig.add_subplot(111)
    vessel_types = ['example_wtiv', 'example_wtiv_us', 'example_heavy_lift_vessel', 'example_ahts_vessel', 'example_feeder']
    init_alloc = [allocs['wtiv'][1][1], allocs['wtiv'][2][1], allocs['wtiv'][0][1], allocs['ahts_vessel'][1], allocs['feeder'][1][1]]
    vessel_count = pd.DataFrame(columns=vessel_types, data = np.ones((len(yrs), len(vessel_types))), index = yrs)
    vessel_count = vessel_count.mul(init_alloc)
    # vessel_count.iloc[0] = init_alloc

    for vessel in vessel_types:
        for vessel_type in futures:
            if vessel_type[1] == vessel:
                years = [x.year for x in vessel_type[2]]
                for year in years:
                    vessel_count.loc[year:,vessel] += 1
    
    # vessel_count.loc[:,'total'] = vessel_count.sum(axis=1)
    # vessel_count['total'] = vessel_count['total'].cumsum()

    return(vessel_count)