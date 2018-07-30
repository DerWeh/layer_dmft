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

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
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
        obj : SpinResolvedArray
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

    def onsite_energy(self, sigma=sigma, hartree=False):
        """Return the single-particle on-site energy.

        The energy is given with respect to half-filling, thus the chemical
        potential μ is corrected by :math:`-U/2`

        Parameters
        ----------
        sigma : {-.5, +5, sigma}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.
        hartree : False or ndarray(float)
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term.

        Returns
        -------
        onsite_energy : float or ndarray(float)
            The (layer dependent) on-site energy :math:`μ + U/2 - V - σh`.

        """
        onsite_energy = np.multiply.outer(sigma, self.h)
        onsite_energy += self.mu + 0.5*self.U - self.V
        if hartree is not False:
            onsite_energy -= hartree * self.U
        return onsite_energy

    def gf0(self, omega, hartree=False, diagonal=True):
        """Return local (diagonal) elements of the non-interacting Green's function.

        Parameters
        ----------
        omega : array(complex)
            Frequencies at which the Green's function is evaluated
        hartree : False or ndarray(float)
            If Hartree Green's function is returned. If it is `False` (default),
            non-interacting Green's function is returned. If it is the electron
            density, the (one-shot) Hartree Green's function is returned.
        diagonal : bool, optional
            Returns only array of diagonal elements if `diagonal` (default).
            Else the whole matrix is returned.

        Returns
        -------
        get_gf_0_loc : SpinResolvedArray(array(complex), array(complex))
            The Green's function for spin up and down.

        """
        sum_str = 'ij, ...j, ji -> i...' if diagonal else 'ij, ...j, jk -> ik...'
        diag = np.diag_indices_from(self.t_mat)
        if hartree is False:  # for common loop
            hartree = (False, False)
        else:  # first axis needs to be spin such that loop is possible
            assert hartree.shape[0] == 2
        gf_0 = {}
        for sp, n in zip(spins, hartree[::-1]):
            gf_0_inv = np.array(self.t_mat, dtype=np.complex256, copy=True)
            gf_0_inv[diag] += self.onsite_energy(sigma=sigma[sp], hartree=n)
            rv_inv, xi, rv = gfmatrix.decompose_gf_omega(gf_0_inv)
            xi_bar = gf.bethe_hilbert_transfrom(omega[..., np.newaxis] + xi,
                                                half_bandwidth=self.D)
            gf_0[sp] = np.einsum(sum_str, rv, xi_bar, rv_inv)

        return SpinResolvedArray(**gf_0)

    def gf_dmft(self, z, self_z, diagonal=True):
        """Return local Green's function for a diagonal self-energy.
        
        This corresponds to the dynamical mean-field theory.

        Parameters
        ----------
        z : ndarray(complex)
            Frequencies at which the Green's function is evaluated.
        self_z : ndarray(complex)
            Self-energy of the green's function. The self-energy is diagonal.
            It's last axis corresponds to the frequencies `z`. The first axis
            contains the spin components and the second the diagonal matrix
            elements.
        diagonal : bool, optional
            Returns only array of diagonal elements if `diagonal` (default).
            Else the whole matrix is returned.

        Returns
        -------
        gf_dmft : SpinResolvedArray
            The Green's function.

        """
        assert z.size == self_z.shape[-1], "Same number of frequencies"
        assert len(spins) == self_z.shape[0], "Two spin components"
        diag = np.diag_indices_from(self.t_mat)
        gf_out = SpinResolvedArray(np.zeros((self_z.size, self_z.size),
                                            dtype=np.complex256))
        for sp, self_sp_z, gf_out_sp in zip(spins, self_z, gf_out):
            gf_0_inv = np.array(self.t_mat, dtype=np.complex256, copy=True)
            gf_0_inv[diag] += self.onsite_energy(sigma=sigma[sp])
            constant = np.diagonal(gf_0_inv).copy()
            for i, zi in enumerate(z):
                gf_0_inv[diag] = constant + zi - self_sp_z[..., i]
                rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_0_inv)
                h_bar = gf.bethe_hilbert_transfrom(h, half_bandwidth=self.D)
                gf_mat = gfmatrix.construct_gf_omega(rv_inv=rv_inv, diag_inv=h_bar, rv=rv)
                gf_out_sp[..., i] = np.diagonal(gf_mat) if diagonal else gf_mat
        return gf_out

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
