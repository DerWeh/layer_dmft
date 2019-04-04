"""Module to provide functionality for charge self-consistent r-DMFT."""
import logging

from functools import partial, partialmethod

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

logging.PROGRESS = logging.INFO - 5  # type: ignore
logging.addLevelName(logging.PROGRESS, 'PROGRESS')  # type: ignore
logging.Logger.progress = partialmethod(logging.Logger.log, logging.PROGRESS)  # type: ignore
logging.progress = partial(logging.log, logging.PROGRESS)  # type: ignore
