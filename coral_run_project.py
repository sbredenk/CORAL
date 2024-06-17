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

    sub_map = pipeline.projects[["name", "substructure"]].set_index("name").to_dict()['substructure']
    df['substructure'] = [sub_map[name] for name in df['name']]
    
    depth_map = pipeline.projects[["name", "depth"]].set_index("name").to_dict()['depth']
    df['depth'] = [depth_map[name] for name in df['name']]

    cod_map = pipeline.projects[["name", "estimated_cod"]].set_index("name").to_dict()['estimated_cod']
    df['estimated_cod'] = [cod_map[name] for name in df['name']]
    df['estimated_cod'] = pd.to_datetime(df['estimated_cod'], format='%Y')

    state_map = pipeline.projects[["name", "location"]].set_index("name").to_dict()['location']
    df['offtake_state'] = [state_map[name] for name in df['name']]

    port_map = pipeline.projects[["name", "associated_port"]].set_index("name").to_dict()['associated_port']
    df['port'] = [port_map[name] for name in df['name']]

    cap_map = pipeline.projects[["name", "capacity"]].set_index("name").to_dict()['capacity']
    df['capacity'] = [cap_map[name] for name in df['name']]

    return manager, df
