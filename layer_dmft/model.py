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
import xarray as xr
import gftools as gt
import gftools.matrix as gtmatrix

from numpy import newaxis

from . import high_frequency_moments as hfm
from .util import spins, Spins, Dimensions as Dim, rev_spin
from .fft import dft_iw2tau


SIGMA = xr.DataArray([0.5, -0.5], dims=[Dim.sp], coords=[list(spins)])
SIGMA.data.flags.writeable = False

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
        self.e_onsite = _ensure_dim(e_onsite, dims=Dim.sp)
        self.U = U
        self.T = T
        self.z = _ensure_dim(z, dims='z')
        self.hybrid_fct = _ensure_dim(hybrid_fct, dims=[Dim.sp, 'z'])
        self.hybrid_mom = _ensure_dim(hybrid_mom, dims=['moment'])

    @property
    def beta(self) -> float:
        """Inverse temperature."""
        return 1./self.T

    @beta.setter
    def beta(self, value: float):
        self.T = 1./value

    def hybrid_tau(self) -> xr.DataArray:
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
        hyb_tau = xr.apply_ufunc(
            lambda hyb, mom: dft_iw2tau(hyb, beta=self.beta, moments=mom),
            self.hybrid_fct, self.hybrid_mom,
            input_core_dims=[[Dim.iws], ['moment']], output_core_dims=[[Dim.tau]]
        )
        hyb_tau.coords[Dim.tau] = np.linspace(0, self.beta, num=hyb_tau.sizes[Dim.tau],
                                              endpoint=True)
        hyb_tau.name = 'Δ'
        return hyb_tau

    def gf0(self, hartree=False) -> xr.DataArray:
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
        e_onsite = self.e_onsite - (0 if hartree is False else hartree*self.U)
        gf_0 = 1./(e_onsite + self.z - self.hybrid_fct)
        return gf_0

    def occ0(self, gf_iw, hartree=False, return_err=True):
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
        e_onsite = self.e_onsite - (hartree*self.U if hartree else 0)
        occ0_ = xr.apply_ufunc(
            partial(gt.density, beta=self.beta, return_err=return_err, matrix=False, total=False),
            gf_iw, -e_onsite,
            input_core_dims=[[Dim.iws], []], output_core_dims=[[], []] if return_err else [],
        )
        if return_err:
            return xr.Dataset({'x': occ0_[0], 'err': occ0_[1]})
        return occ0_

    def gf_s(self, self_z) -> xr.DataArray:
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
        gf_inv = 1./(self.e_onsite + self.z - self_z - self.hybrid_fct)
        return gf_inv


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

    __slots__ = ('_N_l', 'T', 'D', 'params', 't_mat', 'hilbert_transform')

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
        self.params = xr.Dataset(
            {
                'mu': (['layer'], np.zeros(N_l)),
                'V': (['layer'], np.zeros(N_l)),
                'h': (['layer'], np.zeros(N_l)),
                'U': (['layer'], np.zeros(N_l)),
            },
            coords={'layer': range(N_l)}
        )
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
    def beta(self, value: float):
        self.T = 1./value

    @property
    def mu(self) -> xr.DataArray:
        return self.params.mu

    @property
    def V(self) -> xr.DataArray:
        return self.params.V

    @property
    def h(self) -> xr.DataArray:
        return self.params.h

    @property
    def U(self) -> xr.DataArray:
        return self.params.U

    @mu.setter
    def mu(self, value):
        self.params.mu[:] = value

    @V.setter
    def V(self, value):
        self.params.V[:] = value

    @h.setter
    def h(self, value):
        self.params.h[:] = value

    @U.setter
    def U(self, value):
        self.params.U[:] = value

    def onsite_energy(self, sigma=SIGMA, hartree=False) -> xr.DataArray:
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
        params = self.params
        if np.all(params.h == 0):
            try:
                sigma = sigma.mean(keepdims=True)
            except AttributeError:
                sigma = xr.Variable(dims=Dim.sp, data=np.mean(sigma, keepdims=True))
        onsite_energy = sigma*params.h + params.mu + 0.5*params.U - params.V
        if np.any(hartree):
            try:
                hartree.ndim
            except AttributeError:
                hartree = np.asarray(hartree)
            hartree = _ensure_dim(hartree, Dim.lay if hartree.ndim == 1 else [Dim.sp, Dim.lay])
            onsite_energy = onsite_energy - hartree*params.U
        onsite_energy.name = 'onsite energy'
        onsite_energy.attrs['Note'] = 'The onsite energy has the sign of a chemical potential.'
        return onsite_energy

    def hamiltonian(self, sigma=SIGMA, hartree=False) -> xr.DataArray:
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
        t_mat = xr.Variable(dims=['lay1', 'lay2'], data=self.t_mat)
        e_onsite = self.onsite_energy(sigma=sigma, hartree=hartree)
        e_onsite = e_onsite.expand_dims({'lay2': range(self.N_l)}, axis=-1)
        e_onsite = e_onsite.rename(**{Dim.lay: 'lay1'})
        ham = -e_onsite*np.eye(*self.t_mat.shape) - t_mat
        ham.name = 'Hamiltonian'
        return ham

    def gf0(self, omega, hartree=False, diagonal=True) -> xr.DataArray:
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
        omega = _ensure_dim(omega, dims=[f"z_{ii}" for ii in range(np.asanyarray(omega).ndim)])

        def _invert(gf_inv):
            gf_decomp = gtmatrix.decompose_hamiltonian(gf_inv)
            xi_bar = self.hilbert_transform(np.add.outer(gf_decomp.xi, omega.data),
                                            half_bandwidth=self.D)
            return gf_decomp.reconstruct(xi_bar, kind=diag_dic[diagonal])

        layer_dim = [Dim.lay] if diagonal else ['lay1', 'lay2']
        gf0 = xr.apply_ufunc(_invert, gf_0_inv,
                             input_core_dims=[['lay1', 'lay2']],
                             output_core_dims=[[*layer_dim, *omega.dims]],
                             vectorize=True, keep_attrs=True)
        try:
            gf0.coords.update(omega.coords)
        except (TypeError, AttributeError):
            pass
        for lay in layer_dim:
            gf0.coords[lay] = range(self.N_l)
        gf0.name = 'G_{Hartree}' if np.any(hartree) else 'G_0'
        gf0.attrs['temperature'] = self.T
        return gf0

    def gf0_eps(self, eps, omega, hartree=False, diagonal=True):
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
        oadd = np.add.outer
        assert gf_0_inv.ndim == 3
        gf_out = []
        for gf_inv_sp in gf_0_inv:
            gf_decomp = gtmatrix.decompose_hamiltonian(gf_inv_sp)
            xi_bar = 1./oadd(gf_decomp.xi, oadd(-eps, omega))
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
        if total:
            raise NotImplementedError()
        ham = self.hamiltonian(hartree=hartree)
        gf_iw = _ensure_dim(gf_iw, dims=[Dim.sp, Dim.lay, Dim.iws])
        out_dim = [] if total else [Dim.lay]
        dens = xr.apply_ufunc(
            partial(gt.density, beta=self.beta, return_err=return_err, matrix=True, total=total),
            gf_iw, ham,
            input_core_dims=[[Dim.lay, Dim.iws], ['lay1', 'lay2']],
            output_core_dims=[out_dim, out_dim] if return_err else [out_dim],
            vectorize=True, keep_attrs=True,
        )
        if return_err:
            return xr.Dataset({'x': dens[0], 'err': dens[1]})
        return dens

    def occ0_eps(self, eps, hartree=False) -> xr.DataArray:
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
        eps = _ensure_dim(eps, dims='epsilon')

        def _occ0_eps(ham):
            # TODO: vectorize decomposition
            ham_decomp = gtmatrix.decompose_hamiltonian(ham)
            fermi = gt.fermi_fct(np.add.outer(ham_decomp.xi, eps.values), beta=self.beta)
            return ham_decomp.reconstruct(xi=fermi, kind='diag')

        ham = self.hamiltonian(hartree=hartree)
        occ = xr.apply_ufunc(
            _occ0_eps, ham,
            input_core_dims=[['lay1', 'lay2']], output_core_dims=[[Dim.lay, *eps.dims]],
            vectorize=True,
        )
        occ.coords[Dim.lay] = range(self.N_l)
        occ.coords['epsilon'] = eps
        return occ

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

    def gf_dmft_s(self, z, self_z, diagonal=True) -> xr.DataArray:
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
        z = _ensure_dim(z, dims='z')
        gf_inv_diag = self.onsite_energy() + z - self_z
        return self._z_dep_inversion(gf_inv_diag, diagonal=diagonal)

    def gf_dmft_f(self, eff_atom_gf, diagonal=True) -> xr.DataArray:
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
        eff_atom_gf = _ensure_dim(eff_atom_gf, dims=[Dim.lay, 'z'])
        assert len(Spins) == eff_atom_gf.shape[0], "Two spin components"
        return self._z_dep_inversion(1./eff_atom_gf, diagonal=diagonal)

    def gf_dmft_eps_s(self, eps, z, self_z, diagonal=True) -> xr.DataArray:
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
        z = _ensure_dim(z, 'z')
        eps = _ensure_dim(eps, 'epsilon')
        self_z = _ensure_dim(self_z, [Dim.sp, Dim.lay, 'z'])
        diag_z = self.onsite_energy() + z - self_z
        idx = np.eye(self.N_l)

        def _gf(diag):
            mat = diag[:, newaxis]*idx + self.t_mat
            gf_dec = gtmatrix.decompose_gf_omega(mat)
            gf_dec.xi = 1./(gf_dec.xi[..., newaxis] - eps.values)
            return gf_dec.reconstruct(kind=diag_dic[diagonal])

        layer_dim = [Dim.lay] if diagonal else ['lay1', 'lay2']
        gf = xr.apply_ufunc(
            _gf, diag_z,
            input_core_dims=[[Dim.lay]], output_core_dims=[layer_dim + ['epsilon']],
            vectorize=True
        )
        gf.name = 'G'
        gf = gf.transpose(*diag_z.dims[:-2], *layer_dim, *eps.dims, diag_z.dims[-1])
        return gf

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
        idx = np.eye(self.N_l)

        def _inversion(diag):
            mat = diag[:, newaxis]*idx + self.t_mat
            gf_dec = gtmatrix.decompose_gf_omega(mat)
            gf_dec.apply(self.hilbert_transform, half_bandwidth=self.D)
            return gf_dec.reconstruct(kind=diag_dic[diagonal])

        layer_dim = [Dim.lay] if diagonal else ['lay1', 'lay2']
        gf = xr.apply_ufunc(
            _inversion, diag_z,
            input_core_dims=[[Dim.lay]], output_core_dims=[layer_dim],
            vectorize=True, keep_attrs=True
        )
        gf.name = 'G'
        gf = gf.transpose(*diag_z.dims[:-2], *layer_dim, diag_z.dims[-1])
        return gf

    def hybrid_fct_moments(self, occ) -> xr.DataArray:
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
        occ_rev = rev_spin(_ensure_dim(occ, dims=[Dim.sp, Dim.lay]))
        self_mod_0 = self.hamiltonian(hartree=occ_rev)
        self_1 = _diagflat(hfm.self_m1(self.U, occ_rev))
        eps_m2 = self.hilbert_transform.m2(self.D)

        gf_2 = _diagonal(hfm.gf_lattice_m2(self_mod_0))
        gf_3_sub = _diagonal(hfm.gf_lattice_m3_subtract(self_mod_0, eps_m2))
        gf_3 = _diagonal(hfm.gf_lattice_m3(self_mod_0, self_1, eps_m2))
        gf_4_sub = _diagonal(hfm.gf_lattice_m4_subtract(self_mod_0, self_1, eps_m2))

        hyb_m1 = hfm.hybridization_m1(gf_2, gf_3_sub)
        hyb_m2 = hfm.hybridization_m2(gf_2, gf_3, gf_4_sub)

        return xr.concat([hyb_m1, hyb_m2], dim=xr.Variable('moment', ['m1', 'm2']))

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
        lattice = rev_dict_hilbert_transfrom[self.hilbert_transform]
        N_l = self._N_l
        copy = self.__class__(N_l=N_l, lattice=lattice)  # create new object

        for attr in self.__slots__:
            attr_val = getattr(self, attr)
            try:
                attr_val = attr_val.copy(deep=True)
            except (TypeError, AttributeError):
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
        z = _ensure_dim(z, dims='z')
        self_z = _ensure_dim(self_z, dims=[Dim.sp, Dim.lay, 'z'])
        gf_z = (self.gf_dmft_s(z, self_z=self_z, diagonal=True) if gf_z is None
                else _ensure_dim(gf_z, dims=[Dim.sp, Dim.lay, 'z']))
        e_onsite = self.onsite_energy()
        hybrid_z = e_onsite + z - self_z - 1./gf_z
        hybrid_mom = self.hybrid_fct_moments(occ)
        for ll in range(self._N_l):
            lay = {Dim.lay: ll}
            yield SIAM(e_onsite[lay], U=float(self.U[lay]), T=self.T, z=z,
                       hybrid_fct=hybrid_z[lay], hybrid_mom=hybrid_mom[lay])


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


def _ensure_dim(data, dims, coords=None):
    try:
        data.dims
    except AttributeError:
        if coords is None:
            return xr.Variable(dims=dims, data=data)
        return xr.DataArray(data, dims, coords)
    return data


def _diagflat(diagonal: xr.DataArray, diag_dim=Dim.lay, mat_dim=('lay1', 'lay2')
              ) -> xr.DataArray:
    idx = np.eye(diagonal.sizes[diag_dim])
    mat = xr.apply_ufunc(
        lambda dd: dd[..., newaxis]*idx, diagonal,
        input_core_dims=[[diag_dim]], output_core_dims=[list(mat_dim)], keep_attrs=True
    )
    try:
        coord = diagonal.coords[diag_dim]
    except KeyError:
        return mat
    return mat.assign_coords(**{dim: coord.data for dim in mat_dim})


def _diagonal(mat: xr.DataArray, mat_dim=('lay1', 'lay2'), diag_dim=Dim.lay) -> xr.DataArray:
    diagonal = xr.apply_ufunc(
        lambda mat: np.diagonal(mat, axis1=-1, axis2=-2), mat,
        input_core_dims=[list(mat_dim)], output_core_dims=[[diag_dim]], keep_attrs=True
    )
    try:
        coord = mat.coords[mat_dim[0]]
    except KeyError:
        return diagonal
    return diagonal.assign_coords(**{diag_dim: coord.data})



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


def matsubara_frequencies(n_points, beta) -> xr.DataArray:
    iws = xr.DataArray(gt.matsubara_frequencies(n_points, beta=beta),
                       dims=[Dim.iws], coords=[n_points])
    iws.name = 'iω_n'
    iws.attrs['description'] = 'fermionic Matsubara frequencies'
    iws.attrs['temperature'] = 1/beta
    return iws
