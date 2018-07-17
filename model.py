# encoding: utf-8
"""Module to define the layered Hubbard model in use.

The main constituents are:
* The `prm` class which defines the Hamiltonian
  (the layer density of states DOS still needs to be supplemented).
* Spin *objects*: `spins`, `SpinResolved`, `sigma`
  They allow to handle the spin dependence σ=↑=+1/2, σ=↓=−1/2

Most likely you want to import this module like::

    from model import prm, sigma, SpinResolvedArray, spins

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (bytes, input, int, object, open, pow, range, round, str,
                      super, zip)
from collections import namedtuple

import numpy as np

import gftools as gf
import gftools.matrix as gfmatrix

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
    axis with the indices 'up' and 'dn'.

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
                try:
                    return getattr(self, element)
                except TypeError:  # convert string to index and use numpy slicing
                    element = (spins.index(element[0]), ) + element[1:]
                    return super().__getitem__(element)
            except:  # important to raise original error to raise out of range
                raise idx_error

    def __getattr__(self, name):
        """Lazily add the attribute `up`/`dn` and return it."""
        spin_dict = {'up': 0, 'dn': 1}
        if name in spin_dict:  # special cases
            setattr(self, name, self[spin_dict[name]].view(type=np.ndarray))
            return getattr(self, name)
        else:  # default behavior
            raise AttributeError

    @property
    def total(self):
        """Sum of up and down spin."""
        return self.up + self.dn


sigma = SpinResolvedArray(up=0.5, dn=-0.5)
sigma.flags.writeable = False


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

    def onsite_energy(self, sigma=sigma):
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
        try:
            return self.mu + 0.5*self.U - self.V - sigma[:, np.newaxis]*self.h
        except IndexError:  # sigma is a scalar
            return self.mu + 0.5*self.U - self.V - sigma*self.h

    def gf0(self, omega):
        """Return local (diagonal) elements of the non-interacting Green's function.

        Parameters
        ----------
        omega : array(complex)
            Frequencies at which the Green's function is evaluated

        Returns
        -------
        get_gf_0_loc : SpinResolvedArray(array(complex), array(complex))
            The Green's function for spin up and down.

        """
        diag = np.diag_indices_from(self.t_mat)
        gf_0 = {}
        for sp in spins:
            gf_0_inv = np.array(self.t_mat, dtype=np.complex256, copy=True)
            gf_0_inv[diag] += self.onsite_energy(sigma=sigma[sp])
            rv_inv, xi, rv = gfmatrix.decompose_gf_omega(gf_0_inv)
            xi_bar = gf.bethe_hilbert_transfrom(omega[..., np.newaxis] + xi,
                                                half_bandwidth=self.D)
            gf_0[sp] = np.einsum('ij, ...j, ji -> i...', rv, xi_bar, rv_inv)

        return SpinResolvedArray(**gf_0)

    def assert_valid(self):
        """Raise error if attributes are not valid.
        
        Currently only the shape of the parameters is checked.
        """
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
        _str += ", ".join(('{}={!r}'.format(prm, _save_get(prm))
                           for prm in self.__slots__))
        return _str

    def __str__(self):
        def _save_get(attribue):
            try:
                return getattr(self, attribue)
            except AttributeError:
                return '<not assigned>'

        _str = "Hubbard model parameters:\n"
        _str += ",\n ".join(('{}={}'.format(prm, _save_get(prm))
                             for prm in self.__slots__))
        return _str

prm = _Hubbard_Parameters()
