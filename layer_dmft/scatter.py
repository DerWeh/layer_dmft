#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : scatter.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 20.09.2018
# Last Modified Date: 20.09.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
"""Scattering formalism for inhomogeneous many-body problems.

Formulas are mostly from the Byczuk paper on the Friedel sum rule.

"""
import inspect
import logging

from functools import partial

import numba
import numpy as np
import scipy.linalg as la

from numpy import newaxis
import numba.types as nb_t
from scipy import LowLevelCallable
from scipy.integrate import quad
from wrapt import decorator

import gftools as gt
import gftools.numba as gt_numba
from gftools import pade

from .model import rev_dict_hilbert_transfrom, Hubbard_Parameters
from .util import SpinResolvedArray, Spins

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def sorted_eigvals(a, **eigvals_kwds):
    """Sort the eigenvalues of the matrix `a` according to their real part.

    Warnings
    --------
    This is a rather ad-hoc way to establish order, as the eigenvalues are complex
    in general.

    """
    eigvals = la.eigvals(a, **eigvals_kwds)
    sort = eigvals.real.argsort()
    return eigvals[sort]


def layer_phase_fcts(gf_z_in, z_in, n_min: int, n_max: int, valid_z, threshold=1e-8):
    """Return list of functions that return the phase factor for each layer.

    Pade is performed for the eigenvalues, the average is taken after
    calculating the phase.

    TODO: add proper documentation
    """
    coeffs = pade.coefficients(z_in, gf_z_in)

    kind = pade.KindGf(n_min, n_max)
    test_pade = (kind.islice(pade.calc_iterator(z_out=valid_z, z_in=z_in, coeff=coeff_layer))
                 for coeff_layer in coeffs)
    is_valid = [[np.all(pade_.imag < threshold) for pade_ in pade_layer]
                for pade_layer in test_pade]
    n_valid = np.count_nonzero(is_valid, axis=0)
    if np.any(n_valid == 0):
        ll = list(np.argwhere(n_valid == 0))
        raise RuntimeError(f"For layer(s) {ll} no Pade fulfills requirements.")
    elif np.any(n_valid == 1):
        ll = list(np.argwhere(n_valid == 1))

        from warnings import warn

        warn(f"For layer(s) {ll} only one Pade fulfills requirements.\n"
             "It is thus not possible to give a variance.")

    # @jit(nopython=True)
    # def phase(z, pade_z, eps):
    #     number = z - pade_z - eps
    #     return np.arctan2(number.imag, number.real)/np.pi

    # Averager = partial(pade.Mod_Averager, z_in=z_in, mod_fct=phase, kind=kind, vectorized=True)

    def Averager(coeff, valid_pades):
        # TODO: write function putting kind + valid_pades as input to determine
        # Pade array directly -> turn complete averaged except reshaping, gt.Result
        # into jit function

        def averaged(z, eps):
            z = np.asarray(z)
            scalar_input = False
            if z.ndim == 0:
                z = z[np.newaxis]
                scalar_input = True

            pade_iter = kind.islice(pade.calc_iterator(z, z_in, coeff=coeff))
            pades = np.array([pade_ for pade_, valid in zip(pade_iter, valid_pades) if valid])

            number = z - pades - eps
            phase_pi = np.arctan2(number.imag, number.real)
            phase_avg = np.average(phase_pi, axis=0)/np.pi
            std = np.std(phase_pi, axis=0, ddof=1)/np.pi

            if scalar_input:
                return gt.Result(x=np.squeeze(phase_avg), err=np.squeeze(std))
            return gt.Result(x=phase_avg, err=std)
        return averaged

    return [Averager(coeff=coeff_layer, valid_pades=is_valid_layer)
            for coeff_layer, is_valid_layer in zip(coeffs, is_valid)]


def _integrand_function(integrand_function):
    """Wrap `integrand_function` as a `LowLevelCallable` to be used with quad.

    `integrand_function` has to have the signature (float, complex) -> float.

    This speeds up integration by removing call overhead. However only float
    arguments can be passed to and from the function.

    """
    @numba.cfunc(nb_t.float64(nb_t.intc, nb_t.CPointer(nb_t.float64)))
    def wrapped(__, xx):
        return integrand_function(xx[0], xx[1] + xx[2])
    return LowLevelCallable(wrapped.ctypes)


def HomPhaseIntegEps(prm: Hubbard_Parameters, self_z, z_in, n_min, n_max,
                     valid_z, threshold=1e-8):
    r"""Generate function that calculates the ϵ-integrated phase for a homogeneous system.

    .. math:: ϕ(ω) = ∫dϵ DOS(ϵ) \arg G(ϵ, ω)

    Pade is performed for the self-energy + onsite-energy, the average over the
    approximants is taken after calculating and integrating the phase.

    Currently this is only implemented for a single spin and for Bethe DOS...

    FIXME: currently only implemented for h==0

    Parameters
    ----------
    prm : Hubbard_Parameters
        Parameters.
    self_z : (N_sp, N_iw) complex np.ndarray
        Self-energy, first axis corresponds to the spins.
    z_in : (N_iw,) complex np.ndarray
        Input frequencies of the self-energy.
    n_min, n_max : int
        Minimal and maximal number of included Matsubara frequencies.
    valid_z : (M,) complex np.ndarray
        Frequencies at which the imaginary part of the self-energy is checked to
        be positive (thus `valid_z.imag` > 0).
    threshold : float, optional
        Threshold, how negative values are allowed.

    Returns
    -------
    averaged
        Return function (complex) -> Result(float, float) which returns for
        given frequency the ϵ-integrated Phase and its variance from Pade.

    """
    assert np.all(prm.h == 0)
    assert self_z.shape[0] <= 2
    self_z = self_z.mean(axis=0)  # average over spins
    coeffs = pade.coefficients(z_in, self_z)

    # prepare Pade
    kind = pade.KindSelf(n_min, n_max)
    test_pade = kind.islice(pade.calc_iterator(z_out=valid_z, z_in=z_in, coeff=coeffs))
    is_valid = np.array([np.all(pade_.imag < threshold) for pade_ in test_pade])
    assert is_valid.shape[:-1] == self_z.shape[:-1]
    n_valid = np.count_nonzero(is_valid, axis=-1)
    if np.any(n_valid == 0):
        ll = list(np.argwhere(n_valid == 0))
        raise RuntimeError(f"For layer(s) {ll} no Pade fulfills requirements.")
    elif np.any(n_valid == 1):
        LOGGER.warning("For layer(s) %s only one Pade fulfills requirements.\n"
                       "It is thus not possible to give a variance.",
                       list(np.argwhere(n_valid == 1)))

    # prepare numba compiled integral
    assert rev_dict_hilbert_transfrom[prm.hilbert_transform] == 'bethe'
    D = prm.D

    @numba.jit(nopython=True)
    def kernel(eps, z_self_z):
        """Integration kernel."""
        gf_eps_inv = (z_self_z - eps)
        phase = np.arctan2(gf_eps_inv.imag, gf_eps_inv.real)
        return gt_numba.bethe_dos(eps, D)*phase

    lowlevel_kernel = _integrand_function(kernel)
    phase_quad = partial(quad, lowlevel_kernel, a=-prm.D, b=prm.D)

    assert coeffs.ndim == 1 == is_valid.ndim

    def avg_eps_integ_phase(z) -> gt.Result:
        """ϵ-integrated phase from averaged Pade and its variance.

        Parameters
        ----------
        z : complex or complex array_like
            Frequencies at which the analytic continuation is evaluated.

        Returns
        -------
        eps_integ_phase.x, eps_integ_phase.err : complex or complex np.ndarray
            ϵ-integrated phase and the corresponding variance.

        """
        z = np.asarray(z)
        # TODO: check if this is really necessary
        scalar_input = False
        if z.ndim == 0:
            z = z[np.newaxis]
            scalar_input = True

        pade_iter = kind.islice(pade.calc_iterator(z, z_in, coeff=coeffs))
        pades = np.array([pade_ for pade_, valid in zip(pade_iter, is_valid) if valid])

        arg = z - pades
        shape = arg.shape
        phase_pi = np.array([phase_quad(args=(arg_ii.real, arg_ii.imag))[:2]
                             for arg_ii in arg.reshape(-1)])
        phase_pi = phase_pi.reshape(*shape, 2)
        phase_avg = np.average(phase_pi, axis=0)/np.pi
        std = np.std(phase_pi, axis=0, ddof=1)/np.pi
        if scalar_input:
            return gt.Result(x=np.squeeze(phase_avg), err=np.squeeze(std))
        return gt.Result(x=phase_avg, err=std)

    return avg_eps_integ_phase


@decorator
def spin_resolved(wrapped, instance, args, kwds):
    kwargs = inspect.getcallargs(wrapped, *args, **kwds)
    spin_kwargs = {key: arg for key, arg in kwargs.items() if isinstance(arg, SpinResolvedArray)}
    if spin_kwargs:
        bare_args = {key: kwargs[key] for key in set(kwargs) - set(spin_kwargs)}
        res_dict = {sp.name: wrapped(**bare_args, **{key: arg[sp] for key, arg in spin_kwargs.items()})
                    for sp in Spins}
        res_up = res_dict[Spins.up.name]
        if isinstance(res_up, (tuple, list)):
            # if multiple arguments are returned, do SpinResolvedArray for each
            tuple_kind = type(res_up)
            res_dn = res_dict[Spins.dn.name]
            res = tuple_kind(SpinResolvedArray(up=res_up_i, dn=res_dn_i) for
                             res_up_i, res_dn_i in zip(res_up, res_dn))
            return res
        else:
            return SpinResolvedArray(**res_dict)
    else:
        return wrapped(**kwargs)


@spin_resolved
def t_matrix(g_hom, potential):
    r"""Calculate T-matrix from homogeneous Gf and the dynamical potential V.

    The `potential` contains the one-particle scattering potential :math:`V` as
    well es the difference in the self-energy:

    .. math:: \tilde{V}(ω) = V + Σ_{het}(ω) - Σ_{hom}(ω)

    Parameters
    ----------
    g_hom : (N, N) complex ndarray
        The homogeneous Green's function for a specific frequency :math:`ω`.
    potential : (N, N) complex ndarray
        The `potential`. It is in the DMFT approximation diagonal, thus it can
        also be given as a 1-dim array.

    Returns
    -------
    t_matrix : (N, N) complex ndarray
        The T-matrix

    """
    return la.solve(np.eye(*g_hom.shape) - potential@g_hom, potential,
                    overwrite_a=True)


def phase_shift(g_hom, potential):
    r"""Calculate the Friedel sum rule.

    The result is equals to the change in occupation due to potential for a
    Fermi liquid system.

    The formula is

    .. math:: -Tr\arg[1 - V(ω=0)G_{hom}(ω=0)] / π

    Parameters
    ----------
    g_hom : (N, N) complex ndarray
        The homogeneous Green's function for a specific frequency :math:`ω`.
    potential : (N, N) or (N,) complex ndarray
        The `potential`. It is in the DMFT approximation diagonal, thus it can
        also be given as a 1-dim array.

    Returns
    -------
    friedel_sum : float
        The result of the Friedel sum rule.

    """
    kernel = np.eye(*potential.shape) - g_hom@potential
    eigv = la.eigvals(kernel)
    return -np.sum(np.angle(eigv)) / np.pi


class PhaseShiftEps:
    """Calculate ϵ-resolved phase shift."""

    def __init__(self, gf_inv_bare, potential):
        """Prepare `PhaseShiftEps` method to calculate ϵ-resolved Friedel sum.

        Parameters
        ----------
        gf_inv_bare : ([2,] N, N) complex ndarray
            Inverse homogeneous Green's function, where the epsilon dependency
            is striped off.
        potential : ([2,] N, N) or ([2,] N, ) complex ndarray
            The `potential`. It is in the DMFT approximation diagonal, thus it can
            also be given as a 1-dim array.

        """
        self._spin = False
        gf_shape = gf_inv_bare.shape
        assert gf_shape[-1] == gf_shape[-2], "gf_inv_bare needs to be square matrix"
        N = gf_shape[-1]
        self.unity = np.eye(N, N)
        if isinstance(gf_inv_bare, SpinResolvedArray):
            assert gf_inv_bare.ndim == 3, "gf_inv_bare needs to be (2, N, N)"
            assert gf_inv_bare.shape[0] == 2, "2 spin components"
            self._spin = True
        pot_shape = potential.shape
        if isinstance(potential, SpinResolvedArray):
            assert pot_shape[0] == 2, "2 spin components"
            pot_shape = pot_shape[1:]
            self._spin = True
        if len(pot_shape) == 1:
            potential = pot_shape[..., np.newaxis] @ self.unity

        if isinstance(gf_inv_bare, SpinResolvedArray):  # spin case
            dec = {}
            for sp in Spins:
                dec[sp.name] = gt.matrix.decompose_gf_omega(gf_inv_bare[sp])
            self.xi = SpinResolvedArray(up=dec['up'].xi, dn=dec['dn'].xi)
            rv_inv = SpinResolvedArray(up=dec['up'].rv_inv, dn=dec['dn'].rv_inv)
            rv = SpinResolvedArray(up=dec['up'].rv, dn=dec['dn'].rv)
        else:
            dec = gt.matrix.decompose_gf_omega(gf_inv_bare)
            self.xi = dec.xi
            rv, rv_inv = dec.rv, dec.rv_inv

        self.pot_rot = rv_inv @ potential @ rv  # rotated potential

    def __call__(self, eps):
        """Evaluate Friedel sum for given `eps`."""
        eps = np.asarray(eps)[np.newaxis, ..., np.newaxis]  # (N_sp, N_eps, N_l)
        gf = 1./(self.xi[..., np.newaxis, :] - eps)[..., np.newaxis] * self.unity
        kernel = self.unity - gf@self.pot_rot[..., np.newaxis, :, :]

        out_shape = kernel.shape[:-2]
        eigvals = np.array([np.sum(np.angle(la.eigvals(kernel_element)))
                            for kernel_element in kernel.reshape(-1, *kernel.shape[-2:])])
        phi_eps = np.squeeze(eigvals.reshape(out_shape))
        phi_eps = phi_eps.view(type=type(self.xi))
        return -phi_eps/np.pi


class PhaseShiftEps_alt:
    r"""Calculate ϵ-resolved phase shift using alternative formulation.

    .. math:: Φ(ω)
       = sum_i \arg(\frac{1}{ξ^{hom}_i(ω)-ϵ}) - sum_i \arg(\frac{1}{ξ_i(ω)-ϵ})
       = sum_i \arg(ξ_i(ω) - ϵ) - sum_i \arg(ξ^{hom}_i(ω) - ϵ)

    """

    def __init__(self, gf_inv_bar_hom, gf_inv_bar_het):
        self.eigval_hom = la.eigvals(gf_inv_bar_hom)
        self.eigval_het = la.eigvals(gf_inv_bar_het)

    def __call__(self, eps):
        """Evaluate Friedel sum for given `eps`."""
        phi_eps_hom = np.angle(self.eigval_hom - eps[..., newaxis])
        phi_eps_het = np.angle(self.eigval_het - eps[..., newaxis])
        return -np.sum(phi_eps_het - phi_eps_hom, axis=-1)/np.pi
