#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : charge.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 02.08.2018
# Last Modified Date: 07.06.2019
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
import warnings
import logging

from functools import partial, wraps
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy import optimize

import gftools as gt

from . import plot, high_frequency_moments as hfm
from .capacitor_formula import potential_energy_vector
from .util import attribute, Dimensions as Dim
from .model import Hubbard_Parameters

LOGGER = logging.getLogger(__name__)
SMALL_WIDTH = 50


def counter(func):
    """Count how many times function gets executed."""
    @wraps(func)
    def wrapper(*args, **kwds):
        wrapper.count += 1
        return func(*args, **kwds)

    wrapper.count = 0
    return wrapper


@attribute(call=lambda delta_ni: 0)
def get_V(delta_ni):
    """Calculate the mean-field Coulomb potential form Poisson equations.

    The functions needs to be set by `set_getV` before use.

    Parameters
    ----------
    delta_ni : (N_l, ) float np.ndarray
        The difference in occupation with respect to the bulk value.
        Shape is (#layers, ).

    Returns
    -------
    get_V : (N_l, ) float np.ndarray or 0
        The mean-field Coulomb potential.

    """
    return get_V.call(delta_ni)


def set_getV(e_schot=None):
    """Configure the method `get_V` to calculate the mean-field Coulomb potential.

    If `e_schot is None`, the method simply returns `0`, there is no potential.
    Else the potential will be determined using the `potential_energy_vector`
    function.

    Parameters
    ----------
    e_schot : (N_l, ) array_like, optional
        The Schotkey constant, it incorporates lattice constant, electrical
        permeability and other constants.

    """
    if e_schot is None:
        get_V.call = lambda delta_ni: 0
    else:
        e_schot = np.array(e_schot)
        assert e_schot.ndim == 1
        layers = np.arange(e_schot.size)
        get_V.call = partial(potential_energy_vector, e_schot=e_schot, layer_labels=layers)


@counter
def update_occupation(occ_init, i_omega, params, out_dict, self_iw_bare=None):
    r"""Calculate new occupation by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    occ_init : SpinResolvedArray(float, float)
        Start occupation used to calculate new occupation. The expected shape
        is (#spins=2, #layers).
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : model.Hubbard_Parameters
        `prm` object with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.
    out_dict : dict
        Dictionary into which the outputs are written:
        'occ': occupation, 'V': potential and 'Gf': local Green's function

    Returns
    -------
    update_occupation : array(float, float)
        The change in occupation caused by the potential obtained via `get_V`.
        Has the same shape as `occ_init`.

    """
    assert occ_init.shape[0] <= 2
    assert len(occ_init.shape) == 2
    params.V[:] = out_dict['V'] = get_V(occ_init.sum(axis=0) - np.average(occ_init.sum(axis=0)))
    if self_iw_bare is None:
        out_dict['Gf'] = gf_iw = params.gf0(i_omega, hartree=occ_init[::-1])
    else:
        self_iw = self_iw_bare + hfm.self_m0(params.U, occ_init[::-1])[..., np.newaxis]
        gf_iw = params.gf_dmft_s(i_omega, self_z=self_iw)
    occ = out_dict['occ'] = params.occ0(gf_iw, hartree=occ_init[::-1], return_err=False)
    return occ - occ_init


@counter
def update_potential(V_init, i_omega, params, out_dict):
    r"""Calculate new potential by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    V_init : array(float)
            Starting potential used to calculate the occupation and
            subsequently the new potential.
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : model.Hubbard_Parameters
        `model.Hubbard_Parameters` class with the parameters set,
        determining the non-interacting Green's function.
        `params.V` will be updated.
    out_dict : dict
        Dictionary into which the outputs are written:
        'occ': occupation, 'V': potential and 'Gf': local Green's function

    Returns
    -------
    update_potential : array(float)
        The new occupation incorporating the potential obtained via `get_V`

    """
    if np.any(params.U != 0):
        warnings.warn("Only non-interacting case is considered.\n"
                      "Optimize occupation to at least include Hartree term.")
    params.V[:] = V_init
    out_dict['Gf'] = gf_iw = params.gf0(i_omega)
    occ = out_dict['occ'] = params.occ0(gf_iw, return_err=False)
    V = out_dict['V'] = get_V(occ.sum(axis=0) - np.average(occ.sum(axis=0)))
    return V - V_init


@counter
def log_status(x, dx):
    """Intermediate output for `optimize.root`."""
    del x
    if log_status.count == 1:  # print heading in first iteration
        LOGGER.progress('_____________________')
        LOGGER.progress('| iteration | change ')
    LOGGER.progress('| %*i | %s', len('iteration'), log_status.count, np.linalg.norm(dx))


def _root(fun, x0, tol):
    """Wrap root finding.

    From a few test cases `krylov` performs best by far. `broyden1` and
    `df-sane` seem to also be decent options.
    """
    log_status.count = 0
    sol = optimize.root(fun=fun, x0=x0,
                        # method='broyden1',
                        method='krylov',
                        # method='df-sane',
                        tol=tol,
                        # options={'nit': 3},
                        callback=log_status,
                        )
    log_status.count = 0
    return sol


def _occ_least_square(fun, x0, tol, verbose=2):
    """Wrap least square optimization of occupation.

    Least square allows boundaries on the possible values of the occupation.
    It seems to perform more stable, even for hard problems the number of
    function evaluations does not increase too much.
    For simple problems root finding seems to be more efficient.

    To test: use least square for initial estimate (low accuracy) than switch
    to root finding algorithm.
    """
    occ0 = x0
    occ0 = np.asarray(occ0)
    shape = occ0.shape

    def wrapped(x):
        """Accept 1D arrays for compatibility with `least_squares`."""
        x = x.reshape(shape)
        res = (fun(x).reshape(-1))
        return res

    sol = optimize.least_squares(wrapped, x0=occ0.reshape(-1), bounds=(0., 1.),
                                 xtol=tol, method='dogbox', loss='cauchy',
                                 verbose=verbose,)
    return sol


ChargeSelfconsistency = namedtuple('ChargeSelfconsistency', ['sol', 'occ', 'V'])


def charge_self_consistency(parameters, tol, V0=None, occ0=None, kind='auto',
                            n_points=2**11):
    """Charge self-consistency using root-finding algorithm.

    Parameters
    ----------
    parameters : model.Hubbard_Parameters
        `model.Hubbard_Parameters` object with the parameters set,
        determining Hamiltonian.
    tol : float
        Target tol of the self-consistency. Iteration stops if it is
        achieved.
    V0 : ndarray, optional
        Starting value for the electrical potential.
    occ : ndarray, optional
        Starting value for the occupation.
    kind : {'auto', 'occ', 'occ_lsq' 'V'}, optional
        Weather self-consistency is determined according to the charge 'occ' or
        the electrical potential 'V'. In the magnetic case 'V' seems to
        convergence better, there are half as many degrees of freedom. 'occ' on
        the other hand can readily incorporate the Hartree term of the self-energy.
        Per default ('auto') 'occ' is used in the interacting and 'V' in the
        non-interacting case.
        Additionally a constrained (:math:`0 ≤ n_{lσ} ≤ 1`) least-square
        algorithm can be used. If the root-finding algorithms do not converge,
        this `kind` might help.
    n_points : int, optional
        Number of Matsubara frequencies taken into account to calculate the
        charge.

    Returns
    -------
    sol : OptimizedResult
        The solution of the optimization routine, see `optimize.root`.
        `x` contains the actual values, and `success` states if convergence was
        achieved.
    occ : SpinResolvedArray
        The final occupation of the converged result.
    V : ndarray
        The final potential of the converged result.

    """
    # TODO: optimize n_points from error guess
    # TODO: check against given `n` if sum gives right result
    assert kind in set(('auto', 'occ', 'occ_lsq', 'V')), f"Unknown kind: {kind}"
    assert V0 is None or occ0 is None
    params = parameters
    iw_array = xr.Variable(data=gt.matsubara_frequencies(np.arange(n_points), beta=params.beta),
                            dims=Dim.iws)


    if kind == 'auto':
        if np.any(params.U != 0):  # interacting case
            kind = 'occ'  # use 'occ', it can incorporate Hartree term
        else:  # non-interacting case
            kind = 'V'  # use 'V', has half as many parameters to optimize

    if V0 is not None:
        params.V[:] = V0
    output = {}

    # TODO: check tol of density for target tol
    if kind.startswith('occ'):
        if occ0 is None:
            gf_iw = params.gf0(iw_array)
            occ0 = params.occ0(gf_iw, return_err=False)
        optimizer = partial(update_occupation, i_omega=iw_array, params=params, out_dict=output)
        root_finder = _occ_least_square if kind == 'occ_lsq' else _root
        solve = partial(root_finder, fun=optimizer, x0=occ0, tol=tol)
    elif kind == 'V':
        optimizer = partial(update_potential, i_omega=iw_array, params=params, out_dict=output)
        solve = partial(_root, fun=optimizer, x0=params.V[:], tol=tol)
    LOGGER.progress("Search self-consistent occupation number")
    try:
        sol = solve()
    except KeyboardInterrupt as key_err:
        print('Optimization canceled -- trying to continue')
        try:
            hartree_occ = output['occ'][::-1]
        except KeyError:
            print('Failed! No occupation so far.')
            raise key_err
        sol = optimize.OptimizeResult()
        sol.x = output['occ' if kind in ('occ', 'occ_lsq') else 'V']
        sol.success = False
        sol.message = 'Optimization interrupted by user, not terminated.'
    else:
        hartree_occ = output['occ'][::-1]
    # finalize
    gf_iw = params.gf0(iw_array, hartree=hartree_occ)
    occ = params.occ0(gf_iw, hartree=hartree_occ, return_err=True)
    LOGGER.info("Success of finding self-consistent occupation: %s", sol.success)
    LOGGER.info("%s", sol.message)
    return ChargeSelfconsistency(sol=sol, occ=occ, V=params.V)


def charge_self_consistency_int(prm: Hubbard_Parameters, self_iw, occ0, tol):
    """Charge self-consistency using root-finding algorithm.

    Parameters
    ----------
    parameters : model.Hubbard_Parameters
        `model.Hubbard_Parameters` object with the parameters set,
        determining Hamiltonian.
    tol : float
        Target tol of the self-consistency. Iteration stops if it is
        achieved.
    occ : ndarray, optional
        Starting value for the occupation.

    Returns
    -------
    sol : OptimizedResult
        The solution of the optimization routine, see `optimize.root`.
        `x` contains the actual values, and `success` states if convergence was
        achieved.
    occ : SpinResolvedArray
        The final occupation of the converged result.
    V : ndarray
        The final potential of the converged result.

    """
    # TODO: optimize n_points from error guess
    # TODO: check against given `n` if sum gives right result
    # TODO: give back self-energy?
    # optimization: allow mask
    assert self_iw.shape[-2] == occ0.shape[-1] == prm.N_l
    iws = gt.matsubara_frequencies(np.arange(self_iw.shape[-1]), beta=prm.beta)

    # subtract Hartree part
    self_iw_bare = self_iw - hfm.self_m0(prm.U, occ0[::-1])[..., np.newaxis]

    output = {}

    # TODO: check tol of density for target tol
    optimizer = partial(update_occupation, i_omega=iws, params=prm, out_dict=output,
                        self_iw_bare=self_iw_bare)
    solve = partial(_root, fun=optimizer, x0=occ0, tol=tol)
    LOGGER.progress("Search self-consistent occupation number")
    try:
        sol = solve()
    except KeyboardInterrupt as key_err:
        print('Optimization canceled -- trying to continue')
        try:
            hartree_occ = output['occ'][::-1]
        except KeyError:
            print('Failed! No occupation so far.')
            raise key_err
        sol = optimize.OptimizeResult()
        sol.x = output['occ']
        sol.success = False
        sol.message = 'Optimization interrupted by user, not terminated.'
    else:
        hartree_occ = output['occ'][::-1]
    # finalize
    gf_iw = prm.gf_dmft_s(iws, self_z=self_iw_bare + hfm.self_m0(prm.U, hartree_occ)[..., np.newaxis])
    occ = prm.occ0(gf_iw, hartree=hartree_occ, return_err=True)
    LOGGER.info("Success of finding self-consistent occupation: %s", sol.success)
    LOGGER.info("%s", sol.message)
    return ChargeSelfconsistency(sol=sol, occ=occ, V=prm.V)


def plot_results(occ, prm, grid=None):
    """Visualize the results of charge self_consistency quickly."""
    fig, axes = plt.subplots(num='occ + V', nrows=3, ncols=1, sharex=True)
    plot.occ(occ, spin='both', axis=axes[0])
    plot.magnetization(occ, axis=axes[1])
    plot.V(prm, axis=axes[2])
    plt.tight_layout()
    if grid is not None:
        for ax in axes:
            ax.grid(b=grid)
    return fig, axes
    # plt.show()
