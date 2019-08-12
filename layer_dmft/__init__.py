"""Module to provide functionality for charge self-consistent r-DMFT."""
import logging

from functools import partial, partialmethod

# pylint: disable=unused-import
from layer_dmft.model import Hubbard_Parameters, SIAM, hopping_matrix
from layer_dmft._version import get_versions
from layer_dmft.layer_dmft import Runner

__version__ = get_versions()['version']
del get_versions

logging.PROGRESS = logging.INFO - 5  # type: ignore
logging.addLevelName(logging.PROGRESS, 'PROGRESS')  # type: ignore
logging.Logger.progress = partialmethod(logging.Logger.log, logging.PROGRESS)  # type: ignore
logging.progress = partial(logging.log, logging.PROGRESS)  # type: ignore
