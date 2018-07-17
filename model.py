# encoding: utf-8
u"""Module to define the layered Hubbard model in use.

The main constituents are:
* The `prm` class which defines the Hamiltonian
  (the layer density of states DOS still needs to be supplemented).
* Spin *objects*: `spins`, `SpinResolved`, `sigma`
  They allow to handle the spin dependence σ=↑=+1/2, σ=↓=−1/2

Most likely you want to import this module like::

    from model import prm, sigma, SpinResolvedArray, spins

"""
from collections import namedtuple

from builtins import (bytes, input, int, object, open, pow, range, round, str,
                      super, zip)

import numpy as np

spins = ('up', 'dn')


class SpinResolved(namedtuple('Spin', spins)):
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
    with the indices 'up' and 'dn'.

    Attributes
    ----------
    up :
        The up spin component, equal to self[0]
    dn :
        The down spin component, equal to self[1]

    """

    def __new__(cls, *args, **kwargs):
        """Create the object using `np.array` function.

        up, dn : (optional)
            If the keywords `up` *and* `dn` are present, `numpy` uses these
            two parameters to construct the array.

        Returns
        -------
        obj :
            The created `np.ndarray` instance

        """
        try:  # standard initialization via `np.array`
            obj = np.array(*args, **kwargs).view(cls)
        except TypeError:  # alternative: use SpinResolvedArray(up=..., dn=...)
            obj = np.array(object=(kwargs.pop('up'), kwargs.pop('dn')),
                           **kwargs).view(cls)
        assert obj.shape[0] == 2
        obj.up = obj[0].view(type=np.ndarray)
        obj.dn = obj[1].view(type=np.ndarray)
        return obj

    def __getitem__(self, element):
        """Expand `np.ndarray`'s version to handle string indices 'up'/'dn'.

        Regular slices will be handle by numpy, additionally the following can
        be handled:

            1. If the element is in `spins` ('up', 'dn').
            2. If the element's first index is in `spins` and the rest is a
               regular slice. The usage of this is however discouraged.

        """
        try:  # use default np.ndarray method
            return super().__getitem__(element)
        except IndexError as idx_error:  # if element is just ('up'/'dn') use the attribute
            try:
                try:
                    return getattr(self, element)
                except TypeError:  # convert string to index and use numpy slicing
                    element = (spins.index(element[0]), ) + element[1:]
                    return super().__getitem__(element)
            except:  # important to raise original error to raise out of range
                raise idx_error

    @property
    def total(self):
        """Sum of up and down spin."""
        return self.up + self.dn


class _Hubbard_Parameters(object):
    """Parameters of the (layered) Hubbard model.
    
    Attributes
    ----------
    T : float
        temperature
    D : float
        half-bandwidth
    mu : array(float)
        chemical potential of the layers
    V : array(float)
        electrical potential energy
    h : array(float)
        Zeeman magnetic field
    U : array(float)
        onsite interaction
    t_mat : array(float, float)
        hopping matrix

    """

    __slots__ = ('T', 'D', 'mu', 'V', 'h', 'U', 't_mat')

    @property
    def beta(self):
        """Inverse temperature."""
        return 1./self.T

    @beta.setter
    def beta(self, value):
        self.T = 1./value

    def onsite_energy(self, sigma):
        """Return the single-particle on-site energy.

        Parameters
        ----------
        sigma : {-.5, +5}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.

        Returns
        -------
        onsite_energy : float, array(float)
            The (layer dependant) onsite energy :math:`μ + 1/2 U - V - σh`.

        """
        return self.mu + 0.5*self.U - self.V - sigma*self.h

    def assert_valid(self):
        """Raise error if attributes are not valid."""
        if not self.mu.size == self.h.size == self.U.size == self.V.size:
            raise ValueError(
                "all parameter arrays need to have the same shape - "
                "mu: {self.mu.size}, h: {self.h.size}, "
                "U:{self.U.size}, V: {self.V.size}".format(self=self)
            )

    def __repr__(self):
        def _save_get(attribue):
            try:
                return getattr(self, attribue)
            except AttributeError:
                return '<not assigned>'

        _str = "Hubbard model parameters: "
        _str += ", ".join(('{}={}'.format(prm, _save_get(prm))
                           for prm in self.__slots__))
        return _str


sigma = SpinResolvedArray(up=0.5, dn=-0.5)
sigma.flags.writeable = False

prm = _Hubbard_Parameters()
