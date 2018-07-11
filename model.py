# encoding: utf-8
u"""Module to define the layered Hubbard model in use.

The main constituents are:
* The `prm` class which defines the Hamiltonian
  (the layer density of states DOS still needs to be supplemented).
* Spin *objects*: `spins`, `SpinResolved`, `sigma`
  They allow to handle the spin dependence σ=↑=+1/2, σ=↓=−1/2

Most likely you want to import this module like::

    from model import prm, sigma, SpinResolved, spins

"""
from collections import namedtuple

from builtins import (bytes, input, int, object, open, pow, range, round, str,
                      super, zip)

spins = ('up', 'dn')


class SpinResolved(namedtuple('Spin', spins)):
    """Container class for spin resolved quantities.
    
    It is a `namedtuple` which can also be accessed like a `dict`"""
    __slots__ = ()

    def __getitem__(self, element):
        try:
            return super().__getitem__(element)
        except TypeError:
            return getattr(self, element)


sigma = SpinResolved(up=0.5, dn=-0.5)


class _hubbard_model(type):
    """Meta class for `prm` to provide a representation.
    
    TODO: check if meta class should have '__slots__ = ()' for memory.
    """

    def __repr__(self):
        _str = "Hubbard model parameters: "
        _str += ", ".join(('{}={}'.format(prm, getattr(self, prm))
                           for prm in self.__slots__))
        return _str


class prm(object):
    """Parameters of the Hubbard model.
    
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
    __metaclass__ = _hubbard_model  # provides representation

    @classmethod
    def onsite_energy(cls, spin):
        return cls.mu + 0.5*cls.U - cls.V - spin*cls.h

    @classmethod
    def assert_valid(cls):
        """Raise error if attributes are not valid."""
        if not prm.mu.size == prm.h.size == prm.U.size == prm.V.size:
            raise ValueError(
                "all parameter arrays need to have the same shape"
                "mu: {cls.mu.size}, h: {cls.h.size}, "
                "U:{cls.U.size}, V: {cls.V.size}".format(cls)
            )