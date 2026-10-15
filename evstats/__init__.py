"""evstats: extreme value statistics of the halo and galaxy mass distributions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("evstats")
except PackageNotFoundError:  # package is not installed
    __version__ = "unknown"
