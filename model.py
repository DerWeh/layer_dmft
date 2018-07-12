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


prm = _Hubbard_Parameters()
