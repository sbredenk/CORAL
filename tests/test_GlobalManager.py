__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
from copy import deepcopy

import pytest
from ORBIT import ProjectManager

from CORAL import GlobalManager

DIR = os.path.split(__file__)[0]
LIBRARY_PATH = os.path.join(DIR, "test_library")

BASE = config = {
    "wtiv": "example_wtiv",
    "site": {"depth": 20, "distance": 40},
    "plant": {"num_turbines": 20},
    "turbine": "12MW_generic",
    "project_start": 0,
    "design_phases": [],
    "install_phases": ["TurbineInstallation"],
}

BASE_PROJECT = ProjectManager(BASE)
BASE_PROJECT.run()
BASE_PROJECT_TIME = BASE_PROJECT.installation_time


def test_separate_ports():

    allocations = {"port": [("test_port_1", 1), ("test_port_2", 1)]}

    config1 = deepcopy(BASE)
    config1["port"] = "_shared_pool_:test_port_1"

    config2 = deepcopy(BASE)
    config2["port"] = "_shared_pool_:test_port_2"

    configs = [config1, config2]

    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.run()

    assert manager.logs[0]["Finished"] == BASE_PROJECT_TIME
    assert manager.logs[1]["Finished"] == BASE_PROJECT_TIME


def test_shared_ports_limited_resources():

    allocations = {"port": [("test_port_1", 1)]}

    config1 = deepcopy(BASE)
    config1["port"] = "_shared_pool_:test_port_1"

    config2 = deepcopy(BASE)
    config2["port"] = "_shared_pool_:test_port_1"

    configs = [config1, config2]

    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.run()

    assert manager.logs[0]["Finished"] == BASE_PROJECT_TIME
    assert manager.logs[1]["Finished"] == BASE_PROJECT_TIME * 2


def test_shared_ports_sufficient_resources():

    allocations = {"port": [("test_port_1", 2)]}

    config1 = deepcopy(BASE)
    config1["port"] = "_shared_pool_:test_port_1"

    config2 = deepcopy(BASE)
    config2["port"] = "_shared_pool_:test_port_1"

    configs = [config1, config2]

    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.run()

    assert manager.logs[0]["Finished"] == BASE_PROJECT_TIME
    assert manager.logs[1]["Finished"] == BASE_PROJECT_TIME


def test_partial_request_blocking():

    allocations = {
        "wtiv": ("test_wtiv", 2),
        "port": [("test_port_1", 1), ("test_port_2", 1)],
    }

    config1 = deepcopy(BASE)
    config1["wtiv"] = "_shared_pool_:test_wtiv"
    config1["port"] = "_shared_pool_:test_port_1"

    config2 = deepcopy(BASE)
    config2["wtiv"] = "_shared_pool_:test_wtiv"
    config2["port"] = "_shared_pool_:test_port_1"

    config3 = deepcopy(BASE)
    config3["wtiv"] = "_shared_pool_:test_wtiv"
    config3["port"] = "_shared_pool_:test_port_2"

    configs = [config1, config2, config3]

    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.run()

    first, second, third = manager.logs
    assert first["name"] == "Project 1"
    assert second["name"] == "Project 3"
    assert third["name"] == "Project 2"

    assert first["Finished"] == BASE_PROJECT_TIME
    assert second["Finished"] == BASE_PROJECT_TIME
    assert third["Finished"] == 2 * BASE_PROJECT_TIME


def test_simulation_pause():

    allocations = {"port": [("test_port_1", 1)]}
    config1 = deepcopy(BASE)
    config1["port"] = "_shared_pool_:test_port_1"
    config1["project_start"] = 0

    config2 = deepcopy(BASE)
    config2["port"] = "_shared_pool_:test_port_1"
    config2["project_start"] = 5000

    configs = [config1, config2]
    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.run()

    assert len(manager.logs) == 2

    first, second = manager.logs
    assert first["Finished"]
    assert second["Finished"]


def test_staggered_resources_by_int():

    allocations = {
        "wtiv": ("test_wtiv", 2),
        "port": [("test_port_1", 1), ("test_port_2", 1)],
    }

    config = deepcopy(BASE)
    config["wtiv"] = "_shared_pool_:test_wtiv"
    config["port"] = "_shared_pool_:test_port_1"

    configs = [config, config, config]
    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.add_future_resources("wtiv", "test_wtiv", [1000])
    manager.add_future_resources("port", "test_port_1", [400, 800])
    manager.run()

    assert len(manager.logs) == 3

    first, second, third = manager.logs
    assert first["Started"] == 0
    assert second["Started"] == 400
    assert third["Started"] == 1000


def test_zero_initial_resource_capacity():

    allocations = {
        "wtiv": ("test_wtiv", 2),
        "port": [("test_port_1", 1), ("test_port_2", 0)],
    }

    config1 = deepcopy(BASE)
    config1["wtiv"] = "_shared_pool_:test_wtiv"
    config1["port"] = "_shared_pool_:test_port_1"

    config2 = deepcopy(BASE)
    config2["wtiv"] = "_shared_pool_:test_wtiv"
    config2["port"] = "_shared_pool_:test_port_2"

    configs = [config1, config2]
    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)
    manager.add_future_resources("port", "test_port_2", [250])
    manager.run()

    assert len(manager.logs) == 2

    first, second = manager.logs
    assert first["Started"] == 0
    assert second["Started"] == 250


def test_zero_initial_resource_capacity_no_future():

    allocations = {
        "wtiv": ("test_wtiv", 2),
        "port": [("test_port_1", 1), ("test_port_2", 0)],
    }

    config = deepcopy(BASE)
    config["wtiv"] = "_shared_pool_:test_wtiv"
    config["port"] = "_shared_pool_:test_port_2"

    configs = [config]
    manager = GlobalManager(configs, allocations, library_path=LIBRARY_PATH)

    with pytest.raises(RuntimeError):
        manager.run()
