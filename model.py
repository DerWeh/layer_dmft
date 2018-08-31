#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 17.08.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Module to define the layered Hubbard model in use.

The main constituents are:
* The `prm` class which defines the Hamiltonian
  (the layer density of states DOS still needs to be supplemented).
* Spin *objects*: `Spins`, `SpinResolved`, `sigma`
  They allow to handle the spin dependence σ=↑=+1/2, σ=↓=−1/2

Most likely you want to import this module like::

    from model import prm, sigma, Spins

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
from collections import namedtuple
from enum import IntEnum

import numpy as np

import gftools as gf
import gftools.matrix as gfmatrix

spins = ('up', 'dn')


class Spins(IntEnum):
    """Spins 'up'/'dn' with their corresponding index."""
    __slots__ = ()
    up = 0
    dn = 1


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
                    return super().__getitem__(Spins[element])
                except KeyError:  # convert string to index and use numpy slicing
                    element = (Spins[0], ) + element[1:]
                    return super().__getitem__(element)
            except:  # important to raise original error to raise out of range
                raise idx_error

    def __getattr__(self, name):
        """Return the attribute `up`/`dn`."""
        if name in spins:  # special cases
            return self[Spins[name]].view(type=np.ndarray)
        raise AttributeError  # default behavior

    @property
    def total(self):
        """Sum of up and down spin."""
        return self.sum(axis=0)


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

    __slots__ = ('T', 'D', 'mu', 'V', 'h', 'U', 't_mat', 'hilbert_transform')

    def __init__(self):
        """Empty initialization. The assignments are just to help linters."""
        self.T = float
        self.D = float
        self.mu = np.ndarray
        self.V = np.ndarray
        self.h = np.ndarray
        self.U = np.ndarray
        self.t_mat = np.ndarray
        self.hilbert_transform = callable
        for attribute in self.__slots__:
            self.__delattr__(attribute)

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
        sigma : {-0.5, +0.5, sigma}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.
        hartree : False or ndarray(float)
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term. Mind that for the Hartree term
            the spins have to be interchanged.

        Returns
        -------
        onsite_energy : float or ndarray(float)
            The (layer dependent) on-site energy :math:`μ + U/2 - V - σh`.

        """
        onsite_energy = -np.multiply.outer(sigma, self.h)
        onsite_energy += self.mu + 0.5*self.U - self.V
        if hartree is not False:
            assert (len(hartree.shape) == 1
                    if isinstance(sigma, float) else
                    hartree.shape[0] == 2 == len(hartree.shape)), \
                "hartree as no matching shape: {}".format(hartree.shape)
            onsite_energy -= hartree * self.U
        return onsite_energy

    def hamiltonian(self, sigma=sigma, hartree=False):
        """Return the matrix form of the non-interacting Hamiltonian.

        Parameters
        ----------
        sigma : {-0.5, +0.5, sigma}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.
        hartree : False or ndarray(float)
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term.

        Returns
        -------
        hamiltonian : ndarray(float), shape (N, N) or (2, N, N) 
            The Hamiltonian matrix

        """
        ham = -self.onsite_energy(sigma=sigma, hartree=hartree)[..., np.newaxis] \
            * np.eye(*self.t_mat.shape) \
            - self.t_mat
        return ham

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
        for sp, occ in zip(Spins, hartree):
            gf_0_inv = -self.hamiltonian(sigma=sigma[sp], hartree=occ)
            rv_inv, xi, rv = gfmatrix.decompose_hamiltonian(gf_0_inv)
            xi_bar = self.hilbert_transform(omega[..., np.newaxis] + xi,
                                            half_bandwidth=self.D)
            gf_0[sp.name] = np.einsum(sum_str, rv, xi_bar, rv_inv)

        return SpinResolvedArray(**gf_0)

    def occ0(self, gf_iw, hartree=False, return_err=True):
        """Return occupation for the non-interacting (mean-field) model.

        This is a wrapper around `gf.density`.

        Parameters
        ----------
        gf_iw : SpinResolvedArray, shape (2, N, N_matsubara)
            The Matsubara frequency Green's function for positive frequencies
            :math:`iω_n`.  The shape corresponds to the result of `self.gf_0`
            and `self.gf_dmft`.  The last axis corresponds to the Matsubara
            frequencies.
        hartree : False or SpinResolvedArray
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term.
        return_err : bool or float, optional
            If `True` (default), the error estimate will be returned along
            with the density.  If `return_err` is a float, a warning will
            Warning will be issued if the error estimate is larger than
            `return_err`. If `False`, no error estimate is calculated.

        Returns
        -------
        occ0 : SpinResolvedArray, shape (2, N)
            The occupation per layer and spin

        """
        occ0 = {}
        occ0_err = {}
        if hartree is False:
            hartree = (False, False)
        for sp, hartree_sp in zip(Spins, hartree):
            ham = self.hamiltonian(sigma=sigma[sp], hartree=hartree_sp)
            occ0_ = gf.density(gf_iw[sp], potential=-ham, beta=self.beta,
                               return_err=return_err, matrix=True)
            if return_err is True:
                occ0[sp.name], occ0_err[sp.name] = occ0_
            else:
                occ0[sp.name] = occ0_

        if return_err is True:
            return SpinResolvedArray(**occ0), SpinResolvedArray(**occ0_err)
        else:
            return SpinResolvedArray(**occ0)

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
        assert len(Spins) == self_z.shape[0], "Two spin components"
        diag = np.diag_indices_from(self.t_mat)
        if diagonal:
            gf_out = SpinResolvedArray(np.zeros_like(self_z, dtype=np.complex))
        else:
            gf_out = SpinResolvedArray(
                np.zeros((self_z.shape[0], self_z.shape[0], self_z.shape[1]),
                         dtype=np.complex256)
            )
        for sp, self_sp_z, gf_out_sp in zip(Spins, self_z, gf_out):
            gf_0_inv = -self.hamiltonian(sigma=sigma[sp]).astype(np.complex256)
            constant = np.diagonal(gf_0_inv).copy()
            for i, zi in enumerate(z):
                gf_0_inv[diag] = constant + zi - self_sp_z[..., i]
                rv_inv, h, rv = gfmatrix.decompose_gf_omega(gf_0_inv)
                h_bar = self.hilbert_transform(h, half_bandwidth=self.D)
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
        if np.any(self.t_mat.conj().T != self.t_mat):
            raise ValueError(
                "Hamiltonian must be hermitian. "
                "`t_mat`^† = `t_mat` must be fulfilled.\n"
                "t_mat: {t_mat}".format(t_mat=self.t_mat)
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


def chain_hilbert_transform(xi, half_bandwidth=None):
    """Hilbert transform for the isolated 1D chain."""
    return 1./xi

hilbert_transform = {
    'bethe': gf.bethe_hilbert_transfrom,
    'chain': chain_hilbert_transform,
}

prm = _Hubbard_Parameters()
