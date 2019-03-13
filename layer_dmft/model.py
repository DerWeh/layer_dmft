#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 21.01.2019
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Module to define the layered Hubbard model in use.

The main constituents are:
* The `prm` class which defines the Hamiltonian
  (the layer density of states DOS still needs to be supplemented).
* Spin *objects*: `Spins`, `SpinResolved`, `SIGMA`
  They allow to handle the spin dependence σ=↑=+1/2, σ=↓=−1/2

Most likely you want to import this module like::

    from model import prm, SIGMA, Spins

"""
import warnings

from typing import Tuple

import numpy as np
import gftools as gt
import gftools.matrix as gtmatrix

from numpy import newaxis

from .util import SpinResolvedArray, Spins

warnings.simplefilter('always', DeprecationWarning)

SIGMA = SpinResolvedArray(up=0.5, dn=-0.5)
SIGMA.flags.writeable = False

diag_dic = {True: 'diag', False: 'full'}


class SIAM:
    """Single Impurity Anderson model for given frequencies.

    Attributes
    ----------
    see `SIAM.__init__`

    """

    __slots__ = ('T', 'e_onsite', 'U', 'z', 'hybrid_fct', 'hybrid_mom')

    def __init__(self, e_onsite, U: float, T: float, z, hybrid_fct, hybrid_mom) -> None:
        r"""Create effective single impurity Anderson model.

        In frequency space the non-interacting Green's function is

        .. math::
            G_{0\,σ}(z) = (z - ϵ_{σ} - Δ_{σ}(z))^{-1}

        Parameters
        ----------
        e_onsite : (2, ) float array_like
            The onsite energy :math:`ϵ_{σ}` of the impurity.
        U : float
            The local interaction strength of the impurity site.
        T : float
            The temperature.
        z : (N_z, ) complex array_like
            The frequencies for which the hybridization function is given.
        hybrid_fct : (N_z, ) complex array_like
            The hybridization function evaluated at frequencies `z`.
        hybrid_mom : (2, ) float array_like
            The first `1/z` moment of the hybridization function.
            This is necessary to determine the (jump of the) hybridization
            function in :math:`τ`-space correctly.

        """
        self.e_onsite = np.asanyarray(e_onsite)
        self.U = U
        self.T = T
        self.z = z
        self.hybrid_fct = hybrid_fct
        self.hybrid_mom = hybrid_mom
        assert z.size == hybrid_fct.shape[-1]

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return 1./self.T

    @beta.setter
    def beta(self, value):
        self.T = 1./value

    def gf0(self, hartree=False):
        """Return the non-interacting Green's function.

        Parameters
        ----------
        hartree : False or float ndarray
            If Hartree Green's function is returned. If it is `False` (default),
            non-interacting Green's function is returned. If it is the electron
            density, the (one-shot) Hartree Green's function is returned.

        Returns
        -------
        gf_0 : (2, N_z) complex SpinResolvedArray
            The Green's function for spin up and down.

        """
        assert hartree is False, "Not implemented yet"
        if hartree is False:  # for common loop
            hartree = (False, False)
        else:  # first axis needs to be spin such that loop is possible
            assert hartree.shape[0] == 2
        gf_0 = 1./(self.z + self.e_onsite[:, newaxis] - self.hybrid_fct)
        return gf_0.view(type=SpinResolvedArray)

    def occ0(self, gf_iw, hartree=False, return_err=True, total=False):
        """Return occupation for the non-interacting (mean-field) model.

        This is a wrapper around `gt.density`.

        Parameters
        ----------
        gf_iw : (2, N_matsubara) SpinResolvedArray
            The Matsubara frequency Green's function for positive frequencies
            :math:`iω_n`.  The shape corresponds to the result of `self.gf_0`
            and `self.gf_dmft`.  The last axis corresponds to the Matsubara
            frequencies.
        hartree : False or (2, N) SpinResolvedArray
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
        occ0.x : (2, ) SpinResolvedArray
            The occupation per layer and spin
        occ0.err : (2, ) SpinResolvedArray
            If `return_err`, the truncation error of occupation

        """
        assert hartree is False, "Not implemented yet"
        if hartree is False:
            hartree = (False, False)
        # for sp, hartree_sp in zip(Spins, hartree):
        occ0_ = gt.density(gf_iw, potential=-self.e_onsite, beta=self.beta,
                           return_err=return_err, matrix=False, total=total)
        return occ0_

    def gf_s(self, self_z):
        """Calculate the local Green's function from the self-energy `self_z`.

        Parameters
        ----------
        self_z : (2, N) complex ndarray
            Self-energy of the green's function. It's last axis corresponds to
            the frequencies `z`. The first axis contains the spin components.

        Returns
        -------
        gf_z : (2, N) SpinResolvedArray
            The Green's function.

        """
        gf_inv = 1./(self.z + self.e_onsite[:, newaxis] - self_z - self.hybrid_fct)
        return gf_inv.view(type=SpinResolvedArray)


class Hubbard_Parameters:
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

    __slots__ = ('_N_l', 'T', 'D', 'mu', 'V', 'h', 'U', 't_mat', 'hilbert_transform')

    def __init__(self, N_l: int = None) -> None:
        """Empty initialization creating of according shape filled with zeros."""
        self._N_l = N_l
        self.T: float
        self.D: float
        self.hilbert_transform = callable
        del self.hilbert_transform
        if N_l is None:
            warnings.warn("Deprecated, use 'N_l' to state the size.", DeprecationWarning)
            self.mu: np.ndarray
            self.V: np.ndarray
            self.h: np.ndarray
            self.U: np.ndarray
            self.t_mat: np.ndarray
        else:
            self.mu = np.zeros(N_l)
            self.V = np.zeros(N_l)
            self.h = np.zeros(N_l)
            self.U = np.zeros(N_l)
            self.t_mat = np.zeros((N_l, N_l))

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return 1./self.T

    @beta.setter
    def beta(self, value):
        self.T = 1./value

    def onsite_energy(self, sigma=SIGMA, hartree=False):
        """Return the single-particle on-site energy.

        The energy is given with respect to half-filling, thus the chemical
        potential μ is corrected by :math:`-U/2`

        Parameters
        ----------
        sigma : {-0.5, +0.5, sigma}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.
        hartree : False or float ndarray
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term. Mind that for the Hartree term
            the spins have to be interchanged.

        Returns
        -------
        onsite_energy : float or float ndarray
            The (layer dependent) on-site energy :math:`μ + U/2 - V - σh`.

        """
        onsite_energy = +np.multiply.outer(sigma, self.h)
        onsite_energy += self.mu + 0.5*self.U - self.V
        if hartree is not False:
            assert (len(hartree.shape) == 1
                    if isinstance(sigma, float) else
                    len(hartree) == 2 == len(hartree.shape)), \
                f"hartree as no matching shape: {hartree.shape}"
            onsite_energy -= hartree * self.U
        if isinstance(sigma, SpinResolvedArray):
            return onsite_energy.view(type=SpinResolvedArray)
        return onsite_energy

    def hamiltonian(self, sigma=SIGMA, hartree=False):
        """Return the matrix form of the non-interacting Hamiltonian.

        Parameters
        ----------
        sigma : {-0.5, +0.5, SIGMA}
            The value of :math:`σ∈{↑,↓}` which is needed to determine the
            Zeeman energy contribution :math:`σh`.
        hartree : False or float ndarray
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term.

        Returns
        -------
        hamiltonian : (N, N) or (2, N, N) float ndarray
            The Hamiltonian matrix

        """
        ham = -self.onsite_energy(sigma=sigma, hartree=hartree)[..., newaxis] \
            * np.eye(*self.t_mat.shape) \
            - self.t_mat
        return ham

    def gf0(self, omega, hartree=False, diagonal=True):
        """Return local (diagonal) elements of the non-interacting Green's function.

        Parameters
        ----------
        omega : array(complex)
            Frequencies at which the Green's function is evaluated
        hartree : False or float ndarray
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
        if hartree is False:  # for common loop
            hartree = (False, False)
        else:  # first axis needs to be spin such that loop is possible
            assert hartree.shape[0] == 2
        gf_0 = {}
        for sp, occ in zip(Spins, hartree):
            gf_0_inv = -self.hamiltonian(sigma=SIGMA[sp], hartree=occ)
            gf_decomp = gtmatrix.decompose_hamiltonian(gf_0_inv)
            xi_bar = self.hilbert_transform(np.add.outer(gf_decomp.xi, omega),
                                            half_bandwidth=self.D)
            gf_0[sp.name] = gf_decomp.reconstruct(xi_bar, kind=diag_dic[diagonal])

        return SpinResolvedArray(**gf_0)

    def occ0(self, gf_iw, hartree=False, return_err=True, total=False):
        """Return occupation for the non-interacting (mean-field) model.

        This is a wrapper around `gt.density`.

        Parameters
        ----------
        gf_iw : (2, N, N_matsubara) SpinResolvedArray
            The Matsubara frequency Green's function for positive frequencies
            :math:`iω_n`.  The shape corresponds to the result of `self.gf_0`
            and `self.gf_dmft`.  The last axis corresponds to the Matsubara
            frequencies.
        hartree : False or (2, N) SpinResolvedArray
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
        occ0.x : (2, N) SpinResolvedArray
            The occupation per layer and spin
        occ0.err : (2, N) SpinResolvedArray
            If `return_err`, the truncation error of occupation

        """
        occ0 = {}
        occ0_err = {}
        if hartree is False:
            hartree = (False, False)
        for sp, hartree_sp in zip(Spins, hartree):
            ham = self.hamiltonian(sigma=SIGMA[sp], hartree=hartree_sp)
            occ0_ = gt.density(gf_iw[sp], potential=-ham, beta=self.beta,
                               return_err=return_err, matrix=True, total=total)
            if return_err is True:
                occ0[sp.name], occ0_err[sp.name] = occ0_
            else:
                occ0[sp.name] = occ0_

        if return_err is True:
            return gt.Result(x=SpinResolvedArray(**occ0),
                             err=SpinResolvedArray(**occ0_err))
        else:
            return SpinResolvedArray(**occ0)

    def occ0_eps(self, eps, hartree=False):
        r"""Return the :math:`ϵ`-resolved occupation for the non-interacting (mean-field) model.

        `eps` is the dispersion coming from the use of the density of states
        (DOS):

        .. math:: \sum_k → \int dϵ δ(ϵ_k - ϵ)

        This is a wrapper around `gt.density`, there is no error returned as
        the result is exact.

        Parameters
        ----------
        eps : (N_e) float ndarray
            The dispersion parameter :math:`ϵ` at which the density will be
            evaluated.
        hartree : False or (2, N) SpinResolvedArray
            If Hartree term is included. If it is `False` (default) Hartree is
            not included. Else it needs to be the electron density necessary
            to calculate the mean-field term.

        Returns
        -------
        occ0_eps : (2, N, N_e) SpinResolvedArray
            The occupation per layer and spin

        """
        occ0 = {}
        if hartree is False:
            hartree = (False, False)
        for sp, hartree_sp in zip(Spins, hartree):
            ham = self.hamiltonian(sigma=SIGMA[sp], hartree=hartree_sp)
            ham_decomp = gtmatrix.decompose_hamiltonian(ham)
            fermi = gt.fermi_fct(np.add.outer(ham_decomp.xi, eps), beta=self.beta)
            occ0[sp.name] = ham_decomp.reconstruct(xi=fermi, kind='diag')

        return SpinResolvedArray(**occ0)

    # TODO: use spinresolved wrapper. Add option to reverse arguments
    def occ_eps(self, eps, gf_eps_iw, hartree=False, return_err=True, total=False):
        r"""Return the :math:`ϵ`-resolved occupation.

        `eps` is the dispersion coming from the use of the density of states
        (DOS):

        .. math:: \sum_k → \int dϵ δ(ϵ_k - ϵ)

        This is a wrapper around `gt.density`.

        Parameters
        ----------
        eps : (N_e) float ndarray
            The dispersion parameter :math:`ϵ` at which the density will be
            evaluated.
        gf_eps_iw : (2, N_l, N_e, N_iw) SpinResolvedArray
            The Matsubara frequency Green's function for :math:`ϵ` and positive
            frequencies :math:`iω_n`.  The last axis corresponds to the
            Matsubara frequencies.
        hartree : False or (2, N_l) SpinResolvedArray
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
        occ0.x : (2, N_l, N_e) SpinResolvedArray
            The occupation per layer and spin
        occ0.err : (2, N_l, N_e) SpinResolvedArray
            If `return_err`, the truncation error of occupation

        """
        occ = np.zeros(gf_eps_iw.shape[:-1]).view(type=SpinResolvedArray)
        if return_err is True:
            occ_err = np.zeros(gf_eps_iw.shape[:-1]).view(type=SpinResolvedArray)
        if hartree is False:
            hartree = (False, False)
        for sp, hartree_sp in zip(Spins, hartree):
            ham = self.hamiltonian(sigma=SIGMA[sp], hartree=hartree_sp)
            ham_decomp = gtmatrix.decompose_hamiltonian(-ham)
            xi_base = ham_decomp.xi.copy()
            for ii, eps_i in enumerate(eps):
                ham_decomp.xi[:] = xi_base - eps_i
                occ_ = gt.density(
                    gf_eps_iw[sp, ii], potential=ham_decomp, beta=self.beta,
                    matrix=True, return_err=return_err, total=total
                )
                if return_err is True:
                    occ[sp, ..., ii], occ_err[sp, ..., ii] = occ_
                else:
                    occ[sp, ..., ii] = occ_
        if return_err is True:
            return gt.Result(x=occ, err=occ_err)
        else:
            return occ

    def gf_dmft_s(self, z, self_z, diagonal=True):
        """Calculate the local Green's function from the self-energy `self_z`.

        This function is written for the dynamical mean-field theory, where
        the self-energy is diagonal.

        Parameters
        ----------
        z : (N, ) complex ndarray
            Frequencies at which the Green's function is evaluated.
        self_z : (2, N_l, N) complex ndarray
            Self-energy of the green's function. The self-energy is diagonal.
            It's last axis corresponds to the frequencies `z`. The first axis
            contains the spin components and the second the diagonal matrix
            elements.
        diagonal : bool, optional
            Returns only array of diagonal elements if `diagonal` (default).
            Else the whole matrix is returned.

        Returns
        -------
        gf_dmft : (2, N_l, N) SpinResolvedArray
            The Green's function.

        """
        assert z.size == self_z.shape[-1], "Same number of frequencies"
        assert len(Spins) == self_z.shape[0], "Two spin components"
        gf_inv_diag = z + self.onsite_energy()[:, :, newaxis] - self_z
        return self._z_dep_inversion(gf_inv_diag, diagonal=diagonal)

    def gf_dmft_f(self, eff_atom_gf, diagonal=True):
        """Calculate the local Green's function from the effective atomic Gf.

        This function is written for the dynamical mean-field theory, where
        the self-energy is diagonal.

        Parameters
        ----------
        eff_atom_gf : (2, N_l, N) complex ndarray
            The effective atomic Green's function of the impurity problem.
            The first axis corresponds to the spin indices, the second to the
            layers and the last the frequencies `z`.
        diagonal : bool, optional
            Returns only array of diagonal elements if `diagonal` (default).
            Else the whole matrix is returned.

        Returns
        -------
        gf_dmft : (2, N_l, N) SpinResolvedArray
            The Green's function.

        Notes
        -----
        The effective atomic Green's function :math:`F` is defined as

        .. math:: F(z) = [z - ϵ_f - Σ(z)]^{-1},

        where :math:`ϵ_f` is the onsite energy and :math:`Σ` the self-energy of
        the impurity problem.

        """
        assert len(Spins) == eff_atom_gf.shape[0], "Two spin components"
        return self._z_dep_inversion(1./eff_atom_gf, diagonal=diagonal)

    def gf_dmft_eps_s(self, eps, z, self_z, diagonal=True):
        """Calculate the ϵ-dependent Gf from the self-energy `self_z`.

        This function is written for the dynamical mean-field theory, where
        the self-energy is diagonal.

        Parameters
        ----------
        eps : (N_e, ) float ndarray
            Energies :math:`ϵ` at which the Green's function is evaluated.
        z : (N_z, ) complex ndarray
            Frequencies at which the Green's function is evaluated.
        self_z : (2, N_l, N_z) complex ndarray
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
            The Green's function. If `diagonal` the shape is (2, N_l, N_e, N_z),
            else (2, N_l, N_l, N_e, N_z).

        """
        eps = np.asarray(eps)
        assert eps.ndim <= 1
        assert len(self_z.shape) == 3
        assert z.size == self_z.shape[-1]
        shape = self_z.shape
        if diagonal:
            gf_out = SpinResolvedArray(
                np.zeros((shape[0], shape[1], eps.size, shape[2]), dtype=np.complex)
            )
        diag_z = z + self.onsite_energy()[:, :, newaxis] - self_z
        gf_bare_inv = -self.t_mat.astype(np.complex256)
        diag = np.diag_indices_from(gf_bare_inv)
        for diag_z_sp, gf_out_sp in zip(diag_z, gf_out):  # iterate spins
            for ii in range(shape[-1]):  # iterate z-values
                gf_bare_inv[diag] = diag_z_sp[:, ii]
                gf_dec = gtmatrix.decompose_gf_omega(gf_bare_inv)
                gf_dec.xi = 1./(gf_dec.xi[..., newaxis] - eps)
                gf_out_sp[..., ii] = gf_dec.reconstruct(kind=diag_dic[diagonal])
        return gf_out

    def _z_dep_inversion(self, diag_z, diagonal):
        """Calculate Gf from known inverse with diagonal elements `diag_z`.

        The inverse Green's function is given by the diagonal elements `diag_z`
        and the off-diagonal elements `self.t_mat`. The :math:`1*ϵ` part of
        the diagonal is not included and treated separately.

        Parameters
        ----------
        diag_z : (2, N_l, N) complex ndarray
            The diagonal elements of the inverse Green's function, with the
            :math:`ϵ` part stripped. The dimensions are
            (# Spins, # layers, # frequencies).
        diagonal : bool, optional
            Returns only array of diagonal elements if `diagonal` (default).
            Else the whole matrix is returned.

        Returns
        -------
        gf_out : (2, N_l, N) SpinResolvedArray
            The Green's function.

        """
        shape = diag_z.shape
        assert len(shape) == 3  # (# Spin, # Layer, # z)
        if diagonal:
            gf_out = SpinResolvedArray(np.zeros_like(diag_z, dtype=np.complex))
        else:
            gf_out = SpinResolvedArray(
                np.zeros((shape[0], shape[1], shape[1], shape[2]),
                         dtype=np.complex256)
            )
        gf_bare_inv = -self.t_mat.astype(np.complex256)
        diag = np.diag_indices_from(gf_bare_inv)
        # TODO reshape diag_z_sp to see if there is speedup
        for diag_z_sp, gf_out_sp in zip(diag_z, gf_out):  # iterate spins
            for ii in range(shape[-1]):  # iterate z-values
                gf_bare_inv[diag] = diag_z_sp[:, ii]
                gf_dec = gtmatrix.decompose_gf_omega(gf_bare_inv)
                gf_dec.apply(self.hilbert_transform, half_bandwidth=self.D)
                gf_out_sp[..., ii] = gf_dec.reconstruct(kind=diag_dic[diagonal])
        return gf_out

    def hybrid_fct_m1(self, occ):
        r"""Return the first high-frequency moment of the hybridization function.

        The moment is the first order term of the high-frequency expansion of
        the hybridization function :math:`Δ(z)`. It can be obtained

        .. math:: m^{(1)} = \lim_{z → ∞} z Δ(z)

        Parameters
        ----------
        occ : ([2,] N_l) float ndarray
            The local occupation, needed for the constant part of the self-energy.

        Returns
        -------
        m1 : ([2,] N_l) complex ndarray

        """
        hamil = self.hamiltonian(hartree=occ[::-1])
        hoppingterm = {}
        hoppingterm = np.diagonal(hamil @ hamil, axis1=-2, axis2=-1) \
            - np.diagonal(hamil, axis1=-2, axis2=-1)**2
        dos_m2 = self.hilbert_transform.m2(self.D)
        return dos_m2 + hoppingterm

    def assert_valid(self):
        """Raise error if attributes are not valid.

        Currently only the shape of the parameters is checked.
        """
        if not self.mu.size == self.h.size == self.U.size == self.V.size:
            raise ValueError(
                "all parameter arrays need to have the same shape - "
                f"mu: {self.mu.size}, h: {self.h.size}, "
                f"U:{self.U.size}, V: {self.V.size}"
            )
        if np.any(self.t_mat.conj().T != self.t_mat):
            raise ValueError(
                "Hamiltonian must be hermitian. "
                "`t_mat`^† = `t_mat` must be fulfilled.\n"
                f"t_mat: {self.t_mat}"
            )

    def __repr__(self):
        _str = "Hubbard model parameters: "
        _str += ", ".join(f'{prm}={_save_get(self, prm)!r}' for prm in self.__slots__)
        return _str

    def __str__(self):
        _str = "Hubbard model parameters:\n "
        _str += ",\n ".join(f'{prm}={_save_get(self, prm)}' for prm in self.__slots__)
        return _str

    def pstr(self, precision=1):
        """Return pretty string for printing."""
        scalars = ('T', 'D')
        arrays = ('mu', 'V', 'h', 'U')
        width = max(len(el) for el in arrays+scalars)
        _str = "Hubbard model parameters:\n"
        _str += "\n".join(f'{prm:>{width}} = {_save_get(self, prm)}'
                          for prm in scalars) + "\n"
        vals = np.stack([getattr(self, prm) for prm in arrays])
        _str += "\n".join(f'{prm:>{width}} = {value}' for prm, value
                          in zip(arrays, array_printer(vals, precision=precision).split('\n ')))
        _str += "\nt_mat =\n " + array_printer(self.t_mat)
        _str += f"\nhilbert_transform = {rev_dict_hilbert_transfrom[self.hilbert_transform]}"
        _str += "\n"

        return _str

    def __copy__(self):
        copy = self.__class__()  # create new object
        for attr in self.__slots__:
            attr_val = getattr(self, attr)
            try:  # copy the attribute if it provides a copy method
                attr_val = attr_val.copy()
            except AttributeError:  # if not just use it as it is
                pass
            setattr(copy, attr, attr_val)
        return copy

    def copy(self):
        """Return a copy of the Hubbard_Parameters object."""
        return self.__copy__()

    def get_impurity_models(self, z, self_z, gf_z=None, *, occ,
                            only_interacting=False) -> Tuple[SIAM, ...]:
        """Get effective impurity models.

        Parameters
        ----------
        z : (N_z,) complex ndarray
            The frequencies at which the self-energy and Green's function is
            known.
        self_z : (2, N_z) complex ndarray
            The local self-energy of the lattice model.
        gf_z : (2, N_z) complex ndarray, optional
            The local Green's function. If not given, it will be calculated from
            the self-energy `self_z`.
        occ : (2, ) float ndarray
            The occupation corresponding to the self-energy `self_z`. This is
            necessary to calculated the high-frequency (:math:`1/z`) of the
            self-energy and thus the hybridization function.
        only_interacting : bool, optional
            Weather to only calculate the corresponding to interacting layers
            or all (default). For DMFT only the interacting layers are relevant.

        Returns
        -------
        impurity_models : Tuple[SIAM, ...]
            Tuple containing the calculated (see `only_interacting`) single
            impurity Anderson models.

        Examples
        --------
        Get effective SIAMs for interacting layers as starting point for DMFT:

        >>> prm.U = np.array([0. 0., 1., 1.])
        >>> z = gt.matsub
        >>> prm.get_impurity_models()
        >>> N_iw = 2**10
        >>> iw = gt.matsubara_frequencies(N_iw, beta=prm.beta)
        >>> imp_mod_list = prm.get_impurity_models(z=iw, self_z=0, only_interacting=True)

        Identify corresponding layers

        >>> imp_mod_dict = {lay: mod for lay, mod in zip(np.flatnonzero(prm.U), imp_mod_list)}

        """
        if gf_z is None:
            gf_z = self.gf_dmft_s(z, self_z=self_z, diagonal=True)
        e_onsite = self.onsite_energy()
        hybrid_z = z + e_onsite[..., newaxis] - self_z - 1./gf_z
        hybrid_mom = self.hybrid_fct_m1(occ)
        if only_interacting:
            layers = self.U.nonzero()[0]
        else:
            layers = range(self.mu.size)
        impurity_models = tuple(SIAM(e_onsite[:, ll], U=self.U[ll], T=self.T,
                                     z=z, hybrid_fct=hybrid_z[:, ll], hybrid_mom=hybrid_mom[:, ll])
                                for ll in layers)
        return impurity_models


def _save_get(object_, attribue):
    try:
        return getattr(object_, attribue)
    except AttributeError:
        return '<not assigned>'


def array_printer(array, precision=None):
    """Print all elements of the array and strip outermost brackets.

    This function is meant mainly to print 2D arrays.
    """
    string = np.array2string(array, max_line_width=np.infty, threshold=np.infty,
                             precision=precision, suppress_small=True)
    return string[1:-1]  # strip surrounding `[  ]`


def chain_hilbert_transform(xi, half_bandwidth=None):
    """Hilbert transform for the isolated 1D chain."""
    del half_bandwidth  # simply chain has no bandwidth
    return 1./xi


def hopping_matrix(size, nearest_neighbor):
    """Create a hopping matrix with nearest neighbor hopping.

    If `nearest_neighbor` is complex, the lower diagonal will be conjugated to
    ensure hermiticity.
    """
    # TODO: generalize for arbitrary hopping (NN, NNN, ...)
    t_mat = np.zeros((size, size))
    row, col = np.diag_indices(size)
    t_mat[row[:-1], col[:-1]+1] = nearest_neighbor
    t_mat[row[:-1]+1, col[:-1]] = nearest_neighbor.conjugate()
    return t_mat


hilbert_transform = {
    'bethe': gt.bethe_hilbert_transfrom,
    'chain': chain_hilbert_transform,
}
hilbert_transform['bethe'].m2 = gt.bethe_dos.m2
hilbert_transform['chain'].m2 = lambda D: 0

rev_dict_hilbert_transfrom = {transform: name for name, transform
                              in hilbert_transform.items()}


with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    prm = Hubbard_Parameters()
