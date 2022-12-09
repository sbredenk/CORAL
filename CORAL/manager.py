__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import datetime as dt
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
from ORBIT import ProjectManager
from simpy import Event, Environment
from benedict import benedict

from CORAL.library import SharedLibrary


class MultiRequest:
    """Object used to hold multiple simpy.Requests and interface with
    `SharedLibrary` instance."""

    def __init__(self, env, resources, name):
        """
        Creates an instance of `MultiRequest`.

        Parameters
        ----------
        env : simpy.Environment
        resources : dict
        """

        self.trigger = Event(env)
        self.resources = resources
        self.name = name

    def __str__(self) -> str:
        return f"MultiRequest object for {self.name}"


class GlobalManager:
    """Class to manage concurrent ORBIT simulations with shared resources."""

    def __init__(self, configs, allocations, weather=None, library_path=None):
        """
        Creates an instance of `GlobalManager`.

        Parameters
        ----------
        configs : list
            List of ORBIT configurations to run.
        library_path : str
            Path to shared library items.
        allocations : dict
            Number of each library item that exists in the shared environment.
        """

        self._logs = []
        self._projects = {}
        self._counter = Counter()
        self._weather = weather
        self._alloc = allocations
        self.configs = [deepcopy(config) for config in configs]
        self._start = self._get_internal_start_date()

        self.initialize_shared_environment()
        self.library = SharedLibrary(self.env, self._alloc, path=library_path)
        self.setup()

    def _get_internal_start_date(self):
        """Return minimum start_date (int or datetime). Used internally."""
        return min([config["project_start"] for config in self.configs])

    @property
    def logs(self):
        """Return post-processed logs."""

        processed = []
        for log in self._logs:

            new = deepcopy(log)
            if isinstance(self._start, dt.datetime):
                for k in ["Initialized", "Started", "Finished"]:
                    idx = int(np.ceil(log[k]))
                    new[f"Date {k}"] = self._start + dt.timedelta(hours=idx)

            processed.append(new)

        return processed

    @property
    def resource_history(self):
        """Return post-processed resource capacity history."""

        processed = []
        for log in self.library._history:

            new = deepcopy(log)
            new["time"] = self._start + dt.timedelta(hours=log["time"])

            processed.append(new)

        return processed

    def get_results(self, keep_inputs=None):
        """"""

        if keep_inputs is None:
            keep_inputs = []

        df = pd.DataFrame(self.logs).iloc[::-1]
        df = df.reset_index(drop=True).reset_index()

        if keep_inputs:
            # TODO: Add logic for keep_inputs
            pass

        return df

    def initialize_shared_environment(self):
        """Initializes `simpy.Environment` for managing shared resources."""

        self.env = Environment()

    def setup(self):
        """Initialize the projects."""

        for config in self.configs:

            name = self._get_unique_name(config.pop("project_name", "Project"))
            start = config.pop("project_start", 0)

            self.env.process(self._initialize(name, start, benedict(config)))

    def _get_unique_name(self, name):
        """
        Returns a unique name based on instances in `self._counter`.

        Parameters
        ----------
        name : str
            Base name.
        """

        self._counter[name] += 1
        if self._counter[name] == 1 and name != "Project":
            return name

        return f"{name} {self._counter[name]}"

    def run(self):
        """Main simulation run method."""

        self.env.run()
        if len(self.logs) != len(self.configs):
            raise RuntimeError(
                "Missing simulations. Confirm that all resources that started "
                "with zero capacity have future capacity added."
            )

    def _initialize(self, name, start, config):
        """
        Run an individual project configuration.

        Parameters
        ----------
        name : str
            Project handle.
        start : int | float
            Time to initialize project.
        config : dict
            ORBIT configuration.
        """

        idx = self._get_start_idx(start)
        yield self.env.timeout(idx)
        log = {"name": name, "Initialized": self.env.now}

        resources = self._get_shared_resources(config)

        request = MultiRequest(self.env, dict(resources), name)
        resource_data = self.library.request(request)
        yield request.trigger

        log["Started"] = self.env.now
        for key, data in resource_data.items():
            config[key] = data

        project = self._run_project(config)
        yield self.env.timeout(project.project_time)
        log["Finished"] = self.env.now

        self._projects[name] = project
        self._logs.append(log)
        self.library.release(request)

    def _get_start_idx(self, start) -> int:
        """
        Return index of project start time.

        Parameters
        ----------
        start : float | dt.datetime
            Project start time or datetime stamp.
        """

        if isinstance(start, dt.datetime):
            return int(np.ceil((start - self._start).days * 24))

        return int(np.ceil(start))

    @staticmethod
    def _get_shared_resources(config):
        """
        Return shared resources identified in project config.

        Parameters
        ----------
        config : dict
            Initial ORBIT configuration.
        """

        resources = []
        for k, v in config.flatten("//").items():
            try:
                if "_shared_pool_" in v:
                    target = v.split(":")[1]
                    resources.append((k.replace("//", "."), target))

            except TypeError:
                pass

        return resources

    def add_future_resources(self, category, name, dates):
        """
        Schedules future resources to be added to respective resource pool.

        Parameters
        ----------
        category : str
            Resource category.
        name : str
            Resource name.
        dates : list | dt.date
            Date(s) when new resources are added.
        """

        if not isinstance(dates, list):
            dates = [dates]

        for date in dates:

            if isinstance(date, dt.datetime):
                delay = (date - self._start).days * 24

            else:
                delay = date - self._start

            if delay < 0:
                raise ValueError(
                    f"Start date {date} is prior to simulation start."
                )

            self.env.process(self._add_resource(category, name, delay))

    def _add_resource(self, category, name, delay):
        """
        Schedules a new resource to be added to resource pool at time `delay`.

        Parameters
        ----------
        category : str
            Resource category.
        name : str
            Resource name.
        delay : int | float
            Delay time before resource is added to pool.
        """

        yield self.env.timeout(delay)
        self.library.resources[category][name]._capacity += 1
        self.library.check_requests()

    def _run_project(self, config):
        """
        Run configured ORBIT project.

        Parameters
        ----------
        config : dict
            Final ORBIT configuration.
        """

        weather = self._get_current_weather()
        project = ProjectManager(config, weather=weather)
        project.run()

        return project

    @staticmethod
    def _append_request_timing(log, resources, requests):
        """
        Add request timing information to project log.

        Parameters
        ----------
        log : dict
            Project log.
        resources : list
            List of shared resources.
        requests : list
            List of resource requests.
        """

        return {
            **log,
            **{
                f"request-{k}": v.usage_since
                for (k, _), v in zip(resources, requests)
            },
        }

    def _get_current_weather(self):
        """Returns current weather based on `self.env.now`."""

        if self._weather is None:
            return None

        return self._weather[int(np.ceil(self.env.now)) :]
