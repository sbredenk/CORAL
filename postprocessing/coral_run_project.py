from coral_imports import *

def run_manager(pipeline, allocations, library, weather=None, future_resources=None, sorted=False):
    manager = GlobalManager(pipeline.configs, allocations, weather, library_path=library)

    if future_resources != None: 
        for i in future_resources:
            manager.add_future_resources(i[0], i[1], i[2])
        
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
