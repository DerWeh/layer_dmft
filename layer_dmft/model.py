#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : model.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 01.08.2018
# Last Modified Date: 09.05.2019
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
from functools import partial
from typing import Iterable

import numpy as np
import gftools as gt
import gftools.matrix as gtmatrix

from numpy import newaxis

from . import high_frequency_moments as hfm
from .util import SpinResolvedArray, Spins
from .fft import dft_iw2tau


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
        hybrid_fct : (2, N_z) complex array_like
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

    def hybrid_tau(self):
        """Calculate the hybridization function for imaginary times τ in [0, β].

        Returns
        -------
        hybrid_tau : (2, 2*N_z + 1) float np.ndarray
            The Fourier transform of `self.hybrid_fct` on the interval [0, β].

        Raises
        ------
        RuntimeError
            If `self.z` does not correspond to Matsubara frequency, the Fourier
            transform has no defined meaning.

        """
        z = self.z
        if not np.allclose(z, gt.matsubara_frequencies(np.arange(z.size), self.beta)):
            raise RuntimeError("The given frequencies `z` do not correspond to"
                               " the Matsubara frequencies.")
        return dft_iw2tau(self.hybrid_fct, beta=self.beta, moments=self.hybrid_mom)

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
        if hartree is False:
            e_onsite = self.e_onsite
        else:  # first axis needs to be spin such that loop is possible
            e_onsite = self.e_onsite - hartree*self.U
        gf_0 = 1./(self.z + e_onsite[:, newaxis] - self.hybrid_fct)
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

    def __init__(self, N_l: int, lattice: str = None) -> None:
        """Empty initialization creating of according shape filled with zeros."""
        self._N_l = N_l
        self.T: float
        self.D: float
        if lattice is None:
            self.hilbert_transform = callable
            del self.hilbert_transform
            import warnings
            warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
            warnings.warn('Deprecated, state lattice at construction', DeprecationWarning)
        else:
            self.hilbert_transform = hilbert_transform[lattice]
        self.mu = np.zeros(N_l)
        self.V = np.zeros(N_l)
        self.h = np.zeros(N_l)
        self.U = np.zeros(N_l)
        self.t_mat = np.zeros((N_l, N_l))

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return 1./self.T

    @property
    def N_l(self) -> int:
        """Number of layers"""
        return self._N_l

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
        if np.all(self.h == 0):
            sigma = np.mean(sigma, keepdims=True)
        onsite_energy = +np.multiply.outer(sigma, self.h)
        onsite_energy += self.mu + 0.5*self.U - self.V
        if np.any(hartree):
            # assert hartree.ndim <= onsite_energy.ndim
            # backward compatibility
            onsite_energy = onsite_energy - hartree * self.U
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
        gf_0_inv = -self.hamiltonian(hartree=hartree)
        assert gf_0_inv.ndim == 3
        gf_out = []
        for gf_inv_sp in gf_0_inv:
            gf_decomp = gtmatrix.decompose_hamiltonian(gf_inv_sp)
            xi_bar = self.hilbert_transform(np.add.outer(gf_decomp.xi, omega),
                                            half_bandwidth=self.D)
            gf_out.append(gf_decomp.reconstruct(xi_bar, kind=diag_dic[diagonal]))
        return np.array(gf_out).view(type=SpinResolvedArray)

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
        ham = self.hamiltonian(hartree=hartree)
        signature = '(l,w),(l,l)->' + ('({out}),({out})' if return_err else '({out})')
        if total:
            signature = signature.format(out='')
        else:
            signature = signature.format(out='l')
        density = np.vectorize(gt.density, signature=signature,
                               excluded={'beta', 'return_err', 'matrix', 'total'})

        dens = density(gf_iw, -ham, beta=self.beta, return_err=return_err, matrix=True, total=total)
        if return_err:
            return gt.Result(*dens)
        else:
            return dens

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
        assert self_z.ndim == 3
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
        assert diag_z.ndim >= 2  # (..., N_l, #z), typically (#Spin, ...)
        N_l = diag_z.shape[-2]

        gf_bare_inv = -self.t_mat.astype(np.complex256)
        diag_indices = np.diag_indices_from(gf_bare_inv)
        diag_z = np.moveaxis(diag_z, -2, -1)
        newshape = diag_z.shape
        diag_z = diag_z.reshape(-1, N_l)  # (..., N_l)

        out = []
        for diag_el in diag_z:
            gf_bare_inv[diag_indices] = diag_el
            gf_dec = gtmatrix.decompose_gf_omega(gf_bare_inv)
            gf_dec.apply(self.hilbert_transform, half_bandwidth=self.D)
            out.append(gf_dec.reconstruct(kind=diag_dic[diagonal]))
        last_axes = out[0].shape
        gf_out = np.array(out).reshape(newshape[:-1] + last_axes)
        gf_out = np.moveaxis(gf_out, -(len(last_axes) + 1), -1)  # (..., #z)
        return gf_out

    def hybrid_fct_moments(self, occ):
        r"""Return the first high-frequency moments of the hybridization function.

        Currently the first and the second moment are implemented, thus N_moments = 2.

        The moment is the first order term of the high-frequency expansion of
        the hybridization function :math:`Δ(z)`. It can be obtained

        .. math:: m^{(1)} = \lim_{z → ∞} z Δ(z)

        Parameters
        ----------
        occ : ([2,] N_l) float ndarray
            The local occupation, needed for the constant part of the self-energy.

        Returns
        -------
        mom : (N_moments, [2,] N_l) complex ndarray
            Array of the high-frequency moments.

        """
        self_mod_0 = self.hamiltonian(hartree=occ[::-1])
        idx = np.eye(*self_mod_0.shape[-2:])
        self_1 = hfm.self_m1(self.U, occ[::-1])[..., newaxis] * idx
        eps_m2 = self.hilbert_transform.m2(self.D)

        diag = partial(np.diagonal, axis1=-2, axis2=-1)

        gf_2 = diag(hfm.gf_lattice_m2(self_mod_0))
        gf_3_sub = diag(hfm.gf_lattice_m3_subtract(self_mod_0, eps_m2))
        gf_3 = diag(hfm.gf_lattice_m3(self_mod_0, self_1, eps_m2))
        gf_4_sub = diag(hfm.gf_lattice_m4_subtract(self_mod_0, self_1, eps_m2))

        hyb_m1 = hfm.hybridization_m1(gf_2, gf_3_sub)
        hyb_m2 = hfm.hybridization_m2(gf_2, gf_3, gf_4_sub)

        return np.array((hyb_m1, hyb_m2))

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
        # check that vales are assigned
        self.D, self.T  # pylint: disable=pointless-statement

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
        try:
            N_l = self._N_l
        except AttributeError:
            copy = self.__class__()  # create new object
        else:
            copy = self.__class__(N_l=N_l)  # create new object

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

    def get_impurity_models(self, z, self_z, gf_z=None, *, occ) -> Iterable[SIAM]:
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

        Returns
        -------
        impurity_models : Iterable[SIAM, ...]
            Iterable containing the calculated single impurity Anderson models
            for each layer.

        Examples
        --------
        Get effective SIAMs for interacting layers as starting point for DMFT:

        >>> prm.U = np.array([0. 0., 1., 1.])
        >>> N_iw = 2**10
        >>> iw = gt.matsubara_frequencies(N_iw, beta=prm.beta)
        >>> imp_mods = prm.get_impurity_models(z=iw, self_z=0)

        Get interacting layers corresponding layers

        >>> imp_mod_dict = {lay: mod for lay, mod in enumerate(imp_mods) if prm.U[lay] != 0}

        """
        if gf_z is None:
            gf_z = self.gf_dmft_s(z, self_z=self_z, diagonal=True)
        e_onsite = self.onsite_energy()
        hybrid_z = z + e_onsite[..., newaxis] - self_z - 1./gf_z
        hybrid_mom = self.hybrid_fct_moments(occ)
        impurity_models = (SIAM(e_onsite[:, ll], U=self.U[ll], T=self.T,
                                z=z, hybrid_fct=hybrid_z[:, ll], hybrid_mom=hybrid_mom[:, :, ll])
                           for ll in range(self._N_l))
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


def reduce_hubbard(prm: Hubbard_Parameters, mask) -> Hubbard_Parameters:
    """Return `Hubbard_Parameters` containing only the layers in `mask`.

    One of the main use cases for this function is, to calculated the Poisson
    equation on a reduced problem for an improved starting point.

    Parameters
    ----------
    prm : Hubbard_Parameters
        The Hubbard Parameters to reduce.
    mask : slice or array_like
        The mask which will be applied to all array attributes. Depending on
        the fact, whether `mask` generates a copy or view, the attributes of
        the resulting `Hubbard_Parameters` are a copy or a view of the input
        `prm`.

    Returns
    -------
    reduce_hubbard : Hubbard_Parameters
        `Hubbard_Parameters` containing only the layers selected by `mask`.

    """
    mu = prm.mu[mask]
    reduced_prm = Hubbard_Parameters(mu.size,
                                     lattice=rev_dict_hilbert_transfrom[prm.hilbert_transform])
    reduced_prm.T = prm.T
    reduced_prm.D = prm.D
    reduced_prm.mu = mu
    reduced_prm.U = prm.U[mask]
    reduced_prm.h = prm.h[mask]
    reduced_prm.V = prm.V[mask]
    reduced_prm.t_mat = prm.t_mat[mask][:, mask]
    reduced_prm.assert_valid()
    return reduced_prm


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
    'square': gt.square_gf_omega,
}
hilbert_transform['bethe'].m2 = gt.bethe_dos.m2
hilbert_transform['chain'].m2 = lambda D: 0
hilbert_transform['square'].m2 = lambda D: gt.square_dos_moment_coefficients[2]*D**2

rev_dict_hilbert_transfrom = {transform: name for name, transform
                              in hilbert_transform.items()}
