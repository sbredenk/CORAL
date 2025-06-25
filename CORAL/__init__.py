__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Sophie Bredenkamp"
__email__ = "sophie.bredenkamp@nrel.gov"


from .library import SharedLibrary
from .manager import GlobalManager
from .pipelines import Pipeline

from . import _version
__version__ = _version.get_versions()['version']
