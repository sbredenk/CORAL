__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import re
from copy import deepcopy

import numpy as np
import pandas as pd
from geopy import distance
from ORBIT import load_config
import os
import yaml
import time


class Pipeline:
    """Base class for modeling offshore wind project pipelines."""

    def __init__(
        self,
        projects_fp,
        fixed_base_config,
        float_base_config,
        regional_ports=False,
        enforce_feeders=False,
    ):
        """
        Creates an instance of `Pipeline`.

        Parameters
        ----------
        projects_fp : str
            Filepath
        base_config : str
            Filepath
        regional_ports : bool (optional)
            Toggle for regional ports or specific ports.
        enforce_feeders : bool (optional)
            Toggle for enforcing feeder barges for all fixed bottom projects.
        """

        self.projects = pd.read_csv(projects_fp, parse_dates=["start_date"])
        self.append_num_turbines()
        self.base_fixed = load_config(fixed_base_config)
        self.base_float = load_config(float_base_config)
        self.regional_ports = regional_ports
        self.enforce_feeders = enforce_feeders

        self.configs = self.build_configs()


    def append_num_turbines(self):
        """
        Append the number of turbines if missing. Calculated with project and
        turbine capacity.
        """

        if "num_turbines" not in self.projects:
            self.projects["_cap"] = self.projects["turbine"].apply(
                lambda x: float(re.search(r"\d+", x).group(0))
            )
            self.projects["num_turbines"] = (
                self.projects["capacity"] / self.projects["_cap"]
            )
            self.projects["num_turbines"] = self.projects[
                "num_turbines"
            ].apply(lambda x: int(np.ceil(x)))

    def build_configs(self):
        """Iterate through projects in `self.projects` and build ORBIT configs."""

        configs = []
        for _, data in self.projects.iterrows():

            if data["substructure"] == "semisub":
                config = deepcopy(self.base_float)
            else:
                config = deepcopy(self.base_fixed)

            config["project_name"] = data["name"]
            config["project_start"] = data["start_date"]

            config["turbine"] = data["turbine"]
            config["plant"]["num_turbines"] = data["num_turbines"]

            config["site"]["depth"] = data["depth"]

            config["project_coords"] = (data["lat"], data["lon"])
            _calc_dist = self.calculate_port_distance(config, data["associated_port"])
            config["site"]["distance"] = data.get("distance_to_port", _calc_dist)

            config["distance_to_landfall"] = data["distance_to_shore"]

            config["us_wtiv"] = data.get("us_wtiv",False)
            config["us_ffiv"] = data.get("us_ffiv", False)


            if self.regional_ports:
                config["port"] = ":".join(
                    ["_shared_pool_", data["port_region"]]
                )
                # TODO: Check for NaNs in both cases

            else:
                config["port"] = ":".join(
                    ["_shared_pool_", data["associated_port"]]
                )

            config = self.add_substructure_specific_config(
                config, data["substructure"]
            )
            configs.append(config)

        return configs

    def add_substructure_specific_config(self, config, substructure):
        """
        Append substructure specific configurations.

        Parameters
        ----------
        config : dict
            ORBIT config
        substructure : str
            Substructure type
        """
        port = config["port"].replace("_shared_pool_:", "")
        
        if substructure == "monopile":

            # Design Phases
            config["design_phases"] += [
                "MonopileDesign",
                "ScourProtectionDesign",
            ]

            # Install Phases
            config.update(
                    {
                        "install_phases": {
                            "MonopileInstallation": 0
                        }
                    }
                )
            # config["install_phases"]["ScourProtectionInstallation"] = (
            #     "MonopileInstallation",
            #     .25,
            # )
            config["install_phases"]["TurbineInstallation"] = (
                "MonopileInstallation",
                .25,
            )

            # Vessels
            if config["us_wtiv"]:
                config["wtiv"] = "_shared_pool_:example_wtiv_us"
                if port in ["new_bedford"]:
                    config["feeder"] = "_shared_pool_:example_feeder"
                    config["num_feeders"] = 2
            else:
                config["wtiv"] = "_shared_pool_:example_wtiv"
                config["feeder"] = "_shared_pool_:example_feeder"
                config["num_feeders"] = 2

            if port in ["new_bedford", "sbmt", "tradepoint"] or self.enforce_feeders:
                config.update(
                    {
                        "MonopileInstallation": {
                            "wtiv": "_shared_pool_:example_heavy_lift_vessel",
                            "feeder": "_shared_pool_:example_heavy_feeder_1kit",
                            "num_feeders": 2
                        }
                    }
                )
            else:
                config.update(
                    {
                        "MonopileInstallation": {
                            "wtiv": "_shared_pool_:example_heavy_lift_vessel",
                        
                            
                        }
                    }
                )
                

            
        
        elif substructure == "jacket":

            port = config["port"].replace("_shared_pool_:", "") 

            # Design Phases
            config["design_phases"] += [
                "MonopileDesign",
                "ScourProtectionDesign",
            ]

            # Install Phases
            config.update(
                    {
                        "install_phases": {
                            "JacketInstallation": 0
                        }
                    }
                )
            
            # config["install_phases"]["TurbineInstallation"] = 0
            config["install_phases"]["TurbineInstallation"] = (
                "JacketInstallation",
                .25,
            )

            # Vessels
            if config["us_wtiv"]:
                config["wtiv"] = "_shared_pool_:example_wtiv_us"
                if port in ["new_bedford"]:
                    config["feeder"] = "_shared_pool_:example_feeder"
                    config["num_feeders"] = 2
            else:
                config["wtiv"] = "_shared_pool_:example_wtiv"
                config["feeder"] = "_shared_pool_:example_feeder"
                config["num_feeders"] = 2

            if port in ["new_bedford", "sbmt", "tradepoint"]:
                config.update(
                    {
                        "JacketInstallation": {
                            "wtiv": "_shared_pool_:example_heavy_lift_vessel", 
                            "feeder": "_shared_pool_:example_heavy_feeder_1kit", 
                            "num_feeders": 2
                        }
                    }
                )
            else:
                config.update(
                    {
                        "JacketInstallation": {
                            "wtiv": "_shared_pool_:example_heavy_lift_vessel",
                            
                        }
                    }
                )

                

        elif substructure == "gbf":
            # Design Phases
            config["design_phases"] += [
                "MonopileDesign",
                "ScourProtectionDesign",
            ]

            # Install Phases
            config.update(
                    {
                        "install_phases": {
                            "GravityBasedInstallation": 0
                        }
                    }
                )
            

            # Vessels
            config.update(
                {
                    "GravityBasedInstallation": {
                        "ahts_vessel": "_shared_pool_:example_ahts_vessel",
                        "towing_vessel": "_shared_pool_:example_towing_vessel",
                        "towing_vessel_groups": {
                            "towing_vessels": 2,
                            "station_keeping_vessels": 2,
                        },
                        "substructure": {
                            "unit_cost": 0, # placeholder, needed for ORBIT but irrelevant for CORAL
                        }
                    }
                }
            )

            port = config["port"].replace("_shared_pool_:", "")
        
        elif substructure == "semisub":

            # Design Phases
            config["design_phases"] += [
                "SemiSubmersibleDesign",
            ]

            # Install Phases
            config["install_phases"]["MooredSubInstallation"] = 0

            # Vessels
            config.update(
                {
                    "MooredSubInstallation": {
                        "ahts_vessel": "_shared_pool_:example_ahts_vessel",
                        "towing_vessel": "_shared_pool_:example_towing_vessel",
                    },
                }
            )

        else:
            raise TypeError(f"Substructure '{substructure}' not supported.")

        return config

    
    def calculate_port_distance(self, config, port_name):
        
        port_path = os.path.join(os.getcwd(), "analysis", "library", "ports", "%s.yaml" % port_name)
        with open(port_path, 'r') as stream:
            port_data = yaml.safe_load(stream)
        port_coords = (port_data["lat"], port_data["lon"])
        dist = distance.distance(config["project_coords"], port_coords).km

        return(dist)

