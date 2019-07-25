"""Common utilities to interface codes."""
import logging

from collections.abc import Mapping
from subprocess import CalledProcessError, Popen

LOGGER = logging.getLogger(__name__)

class Params(Mapping):
    """Container for Parameters of the code."""

    __slots__ = ()
    __getitem__ = object.__getattribute__
    __setitem__ = object.__setattr__

    def __len__(self):
        """Return the number of attributes in `__slots__`."""
        return len(self.__slots__)

    def __iter__(self):
        """Iterate over attributes in `__slots__`."""
        return iter(self.__slots__)

    @classmethod
    def slots(cls) -> set:
        """Return the set of existing attributes."""
        return set(cls.__slots__)


def execute(command, outfile):
    with open(outfile, "w") as outfile_:
        LOGGER.debug("executing: %s", command)
        proc = Popen(command.split(), stdout=outfile_, stderr=outfile_)
        try:
            proc.wait()
        except:
            proc.kill()
            proc.wait()
            raise
        retcode = proc.poll()
        LOGGER.debug("Code returned with %s", retcode)
        if retcode:
            raise CalledProcessError(retcode, proc.args)
