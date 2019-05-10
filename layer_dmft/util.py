"""Utility classes."""
import sys
import uuid
from pathlib import Path
from enum import IntEnum
from contextlib import contextmanager
from collections import namedtuple
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np

import gftools.pade as gtpade

spins = ('up', 'dn')


class Spins(IntEnum):
    """Spins 'up'/'dn' with their corresponding index.

    Down spins is index -1 (the last one). Thus we can use the same element for
    spin up and down in the paramagnetic case.
    """

    __slots__ = ()
    up = 0
    dn = -1


class SpinResolved(namedtuple('Spin', ('up', 'dn'))):
    """Container class for spin resolved quantities.

    It is a `namedtuple` which can also be accessed like a `dict`
    """

    __slots__ = ()

    def __getitem__(self, element):
        try:
            return super().__getitem__(element)
        except TypeError:
            return getattr(self, element)


class SpinResolvedArray(np.ndarray):
    """Container class for spin resolved quantities allowing array calculations.

    It is a `ndarray` with syntactic sugar. The first axis represents spin and
    thus has to have the dimension 2.
    On top on the typical array manipulations it allows to access the first
    axis with the indices 'up' and 'dn'.

    Attributes
    ----------
    up :
        The up spin component, equal to self[0]
    dn :
        The down spin component, equal to self[1]

    """

    __slots__ = ('up', 'dn')

    def __new__(cls, *args, **kwargs):
        """Create the object using `np.array` function.

        up, dn : (optional)
            If the keywords `up` *and* `dn` are present, `numpy` uses these
            two parameters to construct the array.

        Returns
        -------
        obj : SpinResolvedArray
            The created `np.ndarray` instance

        Raises
        ------
        TypeError
            If the input is neither interpretable as `np.ndarray` nor as up and
            dn spin.

        """
        try:  # standard initialization via `np.array`
            obj = np.array(*args, **kwargs).view(cls)
        except TypeError as type_err:  # alternative: use SpinResolvedArray(up=..., dn=...)
            if {'up', 'dn'} <= kwargs.keys():
                obj = np.array(object=(kwargs.pop('up'), kwargs.pop('dn')),
                               **kwargs).view(cls)
            elif set(Spins) <= kwargs.keys():
                obj = np.array(object=(kwargs.pop(Spins.up), kwargs.pop(Spins.dn)),
                               **kwargs).view(cls)
            else:
                raise TypeError("Invalid construction: " + str(type_err))
        assert obj.shape[0] in (1, 2), "Either values for spin up and dn, or both equal"
        return obj

    def __getitem__(self, element):
        """Expand `np.ndarray`'s version to handle string indices 'up'/'dn'.

        Regular slices will be handle by `numpy`, additionally the following can
        be handled:

            1. If the element is in `spins` ('up', 'dn').
            2. If the element's first index is in `spins` and the rest is a
               regular slice. The usage of this is however discouraged.

        """
        try:  # use default np.ndarray method
            return super().__getitem__(element)
        except IndexError as idx_error:  # if element is just ('up'/'dn') use the attribute
            try:
                if isinstance(element, str):
                    item = super().__getitem__(Spins[element])
                elif isinstance(element, tuple):
                    element = (Spins[element[0]], ) + element[1:]
                    item = super().__getitem__(element)
                else:
                    raise IndexError("Invalid index: ", element)
                return item.view(type=np.ndarray)
            except:  # important to raise original error to raise out of range
                raise idx_error

    def __getattr__(self, name):
        """Return the attribute `up`/`dn`."""
        if name in spins:  # special cases
            return self[Spins[name]].view(type=np.ndarray)
        raise AttributeError(  # default behavior
            f"'{self.__class__.__name__}' object has no attribute {name}"
        )

    @property
    def total(self):
        """Sum of up and down spin."""
        return (self[Spins.up] + self[Spins.dn]).view(type=np.ndarray)


class SelfEnergy(SpinResolvedArray):
    """`ndarray` wrapper for self-energies for the Hubbard model within DMFT."""

    def __new__(cls, input_array, occupation, interaction):
        """Create `SelfEnergy` from existing array_like input.

        Adds capabilities to separate the static Hartree part from the self-
        energy.

        Parameters
        ----------
        input_array : (N_s, N_l, [N_w]) array_like
            Date points for the self-energy, N_s is the number of spins,
            N_l the number of layers and N_w the number of frequencies.
        occupation : (N_s, N_l) array_like
            The corresponding occupation numbers, to calculate the moments.
        interaction : (N_l, ) array_like
            The interaction strength Hubbard :math:`U`.

        """
        obj = np.asarray(input_array).view(cls)
        obj._N_s, obj._N_l = obj.shape[:2]  # #Spins, #Layers
        obj.occupation = np.asanyarray(occupation)
        assert obj.occupation.shape == (obj._N_s, obj._N_l)
        obj.interaction = np.asarray(interaction)
        assert obj.interaction.shape == (obj._N_l, )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.occupation = getattr(obj, 'occupation', None)
        self.interaction = getattr(obj, 'interaction', None)
        self._N_s = getattr(obj, '_N_s', None)
        self._N_l = getattr(obj, '_N_l', None)

    def dynamic(self):
        """Return the dynamic part of the self-energy.

        The static mean-field part is stripped.

        Returns
        -------
        dynamic : (N_s, N_l, [N_w]) ndarray
            The dynamic part of the self-energy

        Raises
        ------
        IndexError
            If the shape of the data doesn't match `occupation` and `interaction`
            shape.

        """
        self: np.ndarray
        if self.shape[:2] != (self._N_s, self._N_l):
            raise IndexError(f"Mismatch of data shape {self.shape} and "
                             f"additional information ({self._N_s}, {self._N_l})"
                             "\n Slicing is not implemented to work with the methods")
        static = self.static(expand=True)
        dynamic = self.view(type=np.ndarray) - static
        # try:
        #     return self - static[..., np.newaxis]
        # except ValueError as val_err:  # if N_w axis doesn't exist
        #     if len(self.shape) != 2:
        #         raise val_err
        #     return self - static
        return dynamic

    def static(self, expand=False):
        """Return the static (Hartree mean-field) part of the self-energy.

        If `expand`, the dimension for `N_w` is added.
        """
        static = self.occupation[::-1] * self.interaction
        if expand and len(self.shape) == 3:
            static = static[..., np.newaxis]
        return static

    def pade(self, z_out, z_in, n_min: int, n_max: int, valid_z=None, threshold=1e-8):
        """Perform Pade analytic continuation on the self-energy.

        Parameters
        ----------
        z_out : complex or array_like
            Frequencies at which the continuation will be calculated.
        z_in : complex ndarray
            Frequencies corresponding to the input self-energy `self`.
        n_min, n_max : int
            Minimum/Maximum number of frequencies considered for the averaging.

        Returns
        -------
        pade.x : (N_s, N_l, N_z_out) ndarray
            Analytic continuation. N_s is the number of spins, N_l the number
            of layer, and N_z_out correspond to `z_in.size`.
        pade.err : (N_s, N_l, N_z_out) ndarray
            The variance corresponding to `pade.x`.

        """
        z_out = np.asarray(z_out)

        # Pade performs better if static part is not stripped from self-energy
        # # static part needs to be stripped as function is for Gf not self-energy
        # self_pade, self_pade_err = pade_fct(self.dynamic())
        kind = gtpade.KindSelf(n_min=n_min, n_max=n_max)
        self_pade = gtpade.avg_no_neg_imag(
            z_out, z_in, fct_z=self, valid_z=valid_z, threshold=threshold, kind=kind
        )
        # return gt.Result(x=self_pade+self.static(expand=True), err=self_pade_err)
        return self_pade


def attribute(**kwds):
    """Add an attribute to a function in a way working with linters."""
    def wrapper(func):
        for key, value in kwds.items():
            setattr(func, key, value)
        return func
    return wrapper


@contextmanager
def local_import(dir_=None):
    """Only import modules within `dir_` (default: cwd)."""
    if dir_ is None:
        dir_ = Path.cwd()
    else:
        dir_ = Path(dir_).absolute().resolve(strict=True)
    import_path0 = sys.path[0]
    sys.path[0] = str(dir_)
    try:
        yield
    finally:
        sys.path[0] = import_path0


def import_file(file, content=None):
    """Try importing `file` as module avoiding name clashes.

    If `content` is given `content = import_file('file.py', 'content')`
    roughly corresponds to `from file import content`
    else `file = import_file('file.py')`
    roughly corresponds to `import file`.

    Parameters
    ----------
    file : str or Path
        The Python file corresponding to the module.
    content : str, optional
        What to import from the module (optional).

    """
    file = Path(file).expanduser().resolve(strict=True)
    spec = spec_from_file_location(file.stem + str(uuid.uuid4()), str(file))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if content:
        return getattr(module, content)
    else:
        return module
