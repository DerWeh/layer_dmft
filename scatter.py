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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)

import inspect

import numpy as np
import scipy.linalg as la

from numpy import newaxis
from wrapt import decorator

import gftools as gt

from model import SpinResolvedArray, Spins


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


class PhaseShiftEps(object):
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


class PhaseShiftEps_alt(object):
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
