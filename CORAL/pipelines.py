__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import re
from copy import deepcopy

import numpy as np
import pandas as pd
from ORBIT import load_config


class BasePipeline:
    """Base class for modeling offshore wind project pipelines."""

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


class FixedPipeline(BasePipeline):
    """
    Class for generating offshore wind project pipelines for fixed bottom East
    Coast projects.
    """

    def __init__(
        self,
        projects_fp,
        base_config,
        sheet_name=0,
        regional_ports=True,
        enforce_feeders=False,
    ):
        """
        Creates an instance of `FixedPipeline`.

        Parameters
        ----------
        projects_fp : str
            Filepath
        base_config : str
            Filepath
        sheet_name : str (optional)
            Sheet name within projects_fp Excel spreadsheet
        regional_ports : bool (optional)
            Toggle for regional ports or specific ports.
        enforce_feeders : bool (optional)
            Toggle for enforcing feeder barges for all fixed bottom projects.
        """

        self.projects = pd.read_csv(projects_fp, parse_dates=["start_date"])
        self.append_num_turbines()
        self.base = load_config(base_config)
        self.regional_ports = regional_ports
        self.enforce_feeders = enforce_feeders

        self.configs = self.build_configs()

    def build_configs(self):
        """Iterate through `self.projects` and build ORBIT configs."""

        configs = []
        for _, data in self.projects.iterrows():

            config = deepcopy(self.base)
            config["project_name"] = data["name"]
            config["project_coords"] = (data["lat"], data["lon"])
            config["project_start"] = data["start_date"]

            config["turbine"] = data["turbine"]
            config["plant"]["num_turbines"] = data["num_turbines"]

            config["site"]["depth"] = data["depth"]
            config["site"]["distance_to_landfall"] = data["distance_to_shore"]

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

        if substructure == "monopile":

            # Design Phases
            config["design_phases"] += [
                "MonopileDesign",
                "ScourProtectionDesign",
            ]

            # Install Phases
            config["install_phases"]["MonopileInstallation"] = 0
            config["install_phases"]["ScourProtectionInstallation"] = (
                "MonopileInstallation",
                1.0,
            )
            # config["install_phases"]["TurbineInstallation"] = 0
            config["install_phases"]["TurbineInstallation"] = (
                "MonopileInstallation",
                1.25,
            )

            # Vessels

            config["wtiv"] = "_shared_pool_:example_wtiv"
            config.update(
                {
                    "MonopileInstallation": {
                        "wtiv": "_shared_pool_:example_heavy_lift_vessel"
                    }
                }
            )

            port = config["port"].replace("_shared_pool_:", "")

            if port in ["sbmt", "new_bedford"] or self.enforce_feeders:
                config["feeder"] = "_shared_pool_:example_feeder"
                config["num_feeders"] = 2

        elif substructure == "jacket":
            raise TypeError("Substructure type 'jacket' not supported.")

        elif substructure == "gbf":
            raise TypeError("Substructure type 'gbf' not supported.")

        else:
            raise TypeError(f"Substructure '{substructure}' not supported.")

        return config


class FloatingPipeline(BasePipeline):
    """
    Class for generating offshore wind project pipelines for floating West
    Coast projects.
    """

    def __init__(self, projects_fp, base_config, sheet_name=0, regional_ports=True):
        """
        Creates an instance of `FixedPipeline`.

        Parameters
        ----------
        projects_fp : str
            Filepath
        base_config : str
            Filepath
        sheet_name : str (optional)
            Sheet name within projects_fp Excel spreadsheet
        regional_ports : bool (optional)
            Toggle for regional ports or specific ports.
        """

        self.projects = pd.read_csv(projects_fp, parse_dates=["start_date"])
        self.append_num_turbines()
        self.base = load_config(base_config)
        self.regional_ports = regional_ports

        self.configs = self.build_configs()

    def build_configs(self):
        """Iterate through `self.projects` and build ORBIT configs."""

        configs = []
        for _, data in self.projects.iterrows():

            config = deepcopy(self.base)
            config["project_name"] = data["name"]
            config["project_start"] = data["start_date"]

            config["turbine"] = data["turbine"]
            config["plant"]["num_turbines"] = data["num_turbines"]

            config["site"]["depth"] = data["depth"]
            config["site"]["distance_to_landfall"] = data["distance_to_shore"]
            config["site"]["distance"] = data["distance_to_site_(km)"]

            if self.regional_ports:
                config["port"] = ":".join(
                    ["_shared_pool_", data["port_region"]]
                )
                # TODO: Check for NaNs in both cases

            else:
                config["port"] = ":".join(
                    ["_shared_pool_", data["associated_port"]]
                )

            # Different install strategies?
            config = self.add_substructure_specific_config(
                config, data["substructure"]
            )

            configs.append(config)

        return configs

    def add_substructure_specific_config(self, config, sub_type):

        if sub_type == "semisub":

            # Design Phases
            config["design_phases"] += [
                "SemiSubmersibleDesign",
                "SemiTautMooringSystemDesign",
            ]

            # Install Phases
            #config["install_phases"]["MooringSystemInstallation"] = 0
            #config["install_phases"]["MooredSubInstallation"] = ('MooringSystemInstallation', 0.5)
            config["install_phases"]["MooredSubInstallation"] = 0

            # Vessels
            #config.update(
            #    {
            #        "MooringSystemInstallation": {
            #            "mooring_install_vessel": "_shared_pool_:example_support_vessel"
            #        }
            #    }
            #)

            config.update(
                {
                    "MooredSubInstallation": {
                        "ahts_vessel": "_shared_pool_:example_ahts_vessel",
                        "towing_vessel": "_shared_pool_:example_towing_vessel",
                    }
                }
            )

        else:
            raise TypeError(f"Substructure type '{sub_type}' not supported.")

        return config
