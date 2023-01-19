__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os

import yaml
from simpy import Resource
from ORBIT.core.library import loader, default_library

CATEGORY_MAP = {"wtiv": "vessels", "feeder": "vessels", "port": "ports"}


class SharedLibrary:
    """Class used to model shared library resources for ORBIT simulations."""

    def __init__(self, env, allocations, path=None):
        """
        Creates an instance of `SharedLibrary`.

        Parameters
        ----------
        path : str
            Path to shared resource library.
        allocations : dict
            Number of each library item that exists in the shared environment.
        """

        self.env = env
        self._alloc = allocations
        self._resources = {}
        self._requests = []
        self._processed = []
        self._history = []

        self.initialize_library_path(path)
        self.initialize_shared_resources()

    @property
    def resources(self):
        """Return dictionary of shared resource objects"""

        return self._resources

    @property
    def resource_list(self):
        """Returns list of shared resource objects."""

        return [
            resource
            for _, data in self.resources.items()
            for _, resource in data.items()
        ]

    @property
    def requests(self):
        """Return active requests."""

        return self._requests

    @property
    def processed_requests(self):
        """Return previously processed requests."""

        return self._processed

    @property
    def unprocessed_requests(self):
        """Return unprocessed requests."""

        return [r for r in self._requests if r not in self._processed]

    def request(self, request):
        """
        Submit a request to the shared library.

        Parameters
        ----------
        request : MultiRequest
        """

        self._requests.append(request)
        self.check_requests()

        _req = {
            k: self.resources[k.split(".")[-1]][v].data
            for k, v in request.resources.items()
        }

        return _req

    def release(self, request):
        """
        Release requests of `SharedResources` associated with MultiRequest
        `request`.

        Parameters
        ----------
        request : MultiRequest
        """

        for k, v in request.resources.items():
            try:
                self.resources[k.split(".")[-1]][v].release(
                    request.requests[k]
                )
            except RuntimeError:
                pass

    def check_requests(self):
        """
        Check unprocessed requests for any projects that can start. This method
        is called as shared resources are released from completed projects.
        """

        for request in self.unprocessed_requests:
            proceed = True

            for k, v in request.resources.items():

                k = k.split(".")[-1]
                if self.resources[k][v].capacity == self.resources[k][v].count:
                    proceed = False
                    break

            if proceed:

                request.requests = {
                    k: self.resources[k.split(".")[-1]][v].request()
                    for k, v in request.resources.items()
                }

                self._processed.append(request)

                try:
                    request.trigger.succeed()

                except RuntimeError:
                    pass

        self._history.append(self.resource_capacities)

    @property
    def resource_capacities(self):

        out = {"time": self.env.now}

        for cat, d in self.resources.items():
            for name, resource in d.items():
                out[f"{cat}:{name}"] = resource.capacity - resource.count

        return out

    def initialize_library_path(self, path):
        """
        Initialize library path at `path` or default ORBIT library is `None`.

        Parameters
        ----------
        path : str
            Path to shared resource library.
        """

        if path is None:
            self._path = default_library

        else:
            self._path = path

    def initialize_shared_resources(self):
        """Initializes shared resources in `self._alloc`."""

        for key, data in self._alloc.items():

            if isinstance(data, tuple):
                data = [data]

            category = CATEGORY_MAP.get(key, "")

            resources = {}
            for name, cap in data:
                path = os.path.join(self._path, category, f"{name}.yaml")

                try:
                    resource = SharedResource(
                        self.env, cap, path, self.check_requests
                    )
                    resources[name] = resource
                except FileNotFoundError:
                    print(
                        f"Warning: Data not found for shared resource '{name}'."
                        f" Verify data file is present at '{path}'"
                    )
                    continue

            self._resources[key] = resources


class SharedResource(Resource):
    """Class to represent a shared set of resources."""

    def __init__(self, env, capacity, path, callback=None):
        """
        Creates an instance of `SharedResource`.

        Parameters
        ----------
        env : Environment
            Environment where shared resources are held.
        capacity : int
            Number of resources in the shared resource set.
        path : str
            Path to library item
        callback : simpy.Event | None
            Function to call on successful resource release.
        """

        if capacity == 0:
            super().__init__(env, 1)
            _ = self.request()

        else:
            super().__init__(env, capacity)

        self.load_data(path)
        self.callback = callback

    def load_data(self, path):
        """Load library data for eventual insert into ORBIT config."""

        with open(path, "r") as f:
            data = yaml.load(f, Loader=loader)

        self.data = data

    def release(self, req):
        """
        Custom release used to trigger upstream recalculation of requests.

        Parameters
        ----------
        req : simpy.Request
            Request to release.
        """

        # TODO: Error handling if request doesn't exist?

        super().release(req)
        if self.callback is not None:
            self.callback()
