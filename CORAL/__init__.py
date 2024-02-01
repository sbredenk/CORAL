__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from . import _version
from .library import SharedLibrary
from .manager import GlobalManager
from .pipelines import FixedPipeline, FloatingPipeline

__version__ = _version.get_versions()["version"]
