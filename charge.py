#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : charge.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 02.08.2018
# Last Modified Date: 16.08.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)

import warnings

from functools import partial, wraps
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

import plot
import gftools as gt
import capacitor_formula

from model import prm, sigma, SpinResolvedArray, spins

VERBOSE = True
SMALL_WIDTH = 50


class _vprint(object):
    __slots__ = ('_printer', )

    def __call__(self, *args, **kwds):
        self._printer(*args, **kwds)


vprint = _vprint()


def verbose_print(func):
    """Decorate `func` to print, only if `func.verbose` or `VERBOSE` if not set.

    The `vprint` in `func` will be executed if `func.verbose` has been set to `True`.
    If `func.verbose` is set to `False` the function is silenced.
    If no attribute `verbose` has been set, the global variable `VERBOSE` is
    used as default.
    """
    def silent_print(*_, **__):
        pass

    print_choice = {
        True: print,
        False: silent_print,
    }

    @wraps(func)
    def wrapper(*args, **kwds):
        try:
            verbose = wrapper.verbose
        except AttributeError:  # no attribute func.verbose
            verbose = VERBOSE
        vprint._printer = print_choice[verbose]
        result = func(*args, **kwds)
        del vprint._printer
        return result
    return wrapper


def counter(func):
    """Counts how many times function gets executed."""
    @wraps(func)
    def wrapper(*args, **kwds):
        wrapper.count += 1
        return func(*args, **kwds)

    wrapper.count = 0
    return wrapper


# FIXME
layers = np.arange(40)
# define get_V
e_schot = np.ones_like(layers) * 0.4
get_V = partial(capacitor_formula.potential_energy_vector,
                e_schot=e_schot, layer_labels=layers)


def self_consistency_plain(parameter, accuracy, mixing=1e-2, n_max=int(1e4)):
    """Naive self-consistency loop using simple mixing."""
    warnings.warn("Don't use plain self-consistency. "
                  "`self_consistency` can also be used with simple mixing",
                  DeprecationWarning)
    params = parameter
    iw_array = gt.matsubara_frequencies(np.arange(int(2**12)), beta=params.beta)

    # start loop paramters
    i, n, n_old = 0, 0, np.infty
    while np.linalg.norm(n - n_old) > accuracy:
        print('**** difference *****')
        print(np.linalg.norm(n - n_old))
        n_old = n
        gf_iw = params.gf0(iw_array)
        n = gt.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
        print('<n>: ', n)
        V_l = get_V(n.up + n.dn - np.average(n))  # FIXME: better dimension checks!!!
        print('V_l: ', V_l)
        params.V[:] = mixing * V_l + (1-mixing)*params.V
        print('V_l mixed: ', params.V[:])
        print(i)
        i += 1
        if i > n_max:
            print('maximum reached')
            break
    print('Final occupation:')
    print(n)
    print('Final potential')
    print(V_l)
    print(prm.V)
    print(np.linalg.norm(prm.V - V_l))


@counter
def update_occupation(n_start, i_omega, params, out_dict):
    r"""Calculate new occupation by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    n_start : SpinResolvedArray(float, float)
        Start occupation used to calculate new occupation. The expected shape
        is (#spins=2, #layers).
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : prm
        `prm` object with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.
    out_dict : dict
        Dictionary into which the outputs are written:
        'occ': occupation, 'V': potential and 'Gf': local Green's function

    Returns
    -------
    update_occupation : array(float, float)
        The change in occupation caused by the potential obtained via `get_V`.
        Has the same shape as `n_start`.

    """
    assert n_start.shape[0] == 2
    assert len(n_start.shape) == 2
    params.V[:] = out_dict['V'] = get_V(n_start.sum(axis=0) - np.average(n_start.sum(axis=0)))
    gf_iw = out_dict['Gf'] = params.gf0(i_omega, hartree=n_start)
    n = out_dict['occ'] = gt.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)[0]
    return n - n_start


@counter
def update_potential(V_start, i_omega, params, out_dict):
    r"""Calculate new potential by setting :math:`V_l` from the method `get_V`.

    This methods modifies `params.V`.

    Parameters
    ----------
    V_start : array(float)
            Starting potential used to calculate the occupation and
            subsequently the new potential.
    i_omega : array(complex)
        Matsubara frequencies :math:`iω_n` at which the Green's function is
        evaluated to calculate the occupation.
    params : prm
        `prm` class with the parameters set, determining the non-interacting
        Green's function. `params.V` will be updated.
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
    params.V[:] = V_start
    out_dict['Gf'] = gf_iw = params.gf0(i_omega)
    n = out_dict['occ'] = gt.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)[0]
    V = out_dict['V'] = get_V(n.sum(axis=0) - np.average(n.sum(axis=0)))
    return V - V_start


@counter
def print_status(x, dx):
    """Intermediate print output for `optimize.root`."""
    print("======== " + str(print_status.count) + " =============")
    # print(x)
    print("--- " + str(np.linalg.norm(dx)) + " ---")


def _occ_root(fun, occ0, tol, verbose=True):
    """Wrapper for root finding for occupation.
    
    From a few test cases `krylov` performs best by far. `broyden1` and
    `df-sane` seem to also be decent options.
    """
    sol = optimize.root(fun=fun, x0=occ0,
                        # method='broyden1',
                        method='krylov',
                        # method='df-sane',
                        tol=tol,
                        # options={'nit': 3},
                        callback=print_status if verbose else None,
                        )
    return sol


def _occ_least_square(fun, occ0, tol, verbose=2):
    """Wrapper for least square optimization of occupation.
    
    Least square allows boundaries on the possible values of the occupation.
    It seems to perform more stable, even for hard problems the number of
    function evaluations does not increase too much.
    For simple problems root finding seems to be more efficient.

    To test: use least square for initial estimate (low accuracy) than switch 
    to root finding algorithm.
    """
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


def _pot_root(fun, pot0, tol, verbose=True):
    """Wrapper for root finding of potential."""
    sol = optimize.root(fun=fun, x0=pot0,
                        # method='broyden1',
                        method='krylov',
                        # method='df-sane',
                        tol=tol,
                        # options={'nit': 3},
                        callback=print_status if verbose else None,
                        )
    return sol


ChargeSelfconsistency = namedtuple('ChargeSelfconsistency', ['sol', 'occ', 'V'])


@verbose_print
def charge_self_consistency(parameters, accuracy, V0=None, kind='auto'):
    """Charge self-consistency using root-finding algorithm.

    Parameters
    ----------
    parameters : prm
        `prm` object with the parameters set, determining Hamiltonian.
    accuracy : float
        Target accuracy of the self-consistency. Iteration stops if it is
        achieved.
    V0 : ndarray, optional
        Starting value for the occupation or electrical potential.
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
    # TODO: check against given `n` if sum gives right result
    assert kind in set(('auto', 'occ', 'occ_lsq', 'V')), "Unknown kind: {}".format(kind)
    params = parameters
    iw_array = gt.matsubara_frequencies(np.arange(int(2**10)), beta=params.beta)

    if kind == 'auto':
        if np.any(params.U != 0):  # interacting case
            kind = 'occ'  # use 'occ', it can incorporate Hartree term
        else:  # non-interacting case
            kind = 'V'  # use 'V', has half as many parameters to optimize

    if V0 is not None:
        params.V[:] = V0
    output = {}

    vprint('optimize'.center(SMALL_WIDTH, '='))
    # TODO: check accuracy of density for target accuracy
    if kind == 'occ':
        gf_iw = params.gf0(iw_array)
        x0, __ = gt.density(gf_iw, params.onsite_energy(),
                            beta=params.beta)
        optimizer = partial(update_occupation, i_omega=iw_array, params=params,
                            out_dict=output)
        sol = _occ_root(optimizer, occ0=x0, tol=accuracy, verbose=True)
    elif kind == 'occ_lsq':
        gf_iw = params.gf0(iw_array)
        x0, __ = gt.density(gf_iw, params.onsite_energy(),
                            beta=params.beta)
        optimizer = partial(update_occupation, i_omega=iw_array, params=params,
                            out_dict=output)
        sol = _occ_least_square(optimizer, occ0=x0, tol=accuracy, verbose=2)
    elif kind == 'V':
        if V0 is None:
            V0 = np.zeros_like(parameters.mu)
            params.V[:] = V0
        optimizer = partial(update_potential, i_omega=iw_array, params=params,
                            out_dict=output)
        sol = _pot_root(optimizer, pot0=V0, tol=accuracy, verbose=True)
    vprint("".center(SMALL_WIDTH, '='))
    vprint("Success: {opt.success}".format(opt=sol))
    vprint('optimized paramter'.center(SMALL_WIDTH, '-'))
    vprint(sol.x)
    vprint('final potential'.center(SMALL_WIDTH, '-'))
    vprint(params.V)
    vprint("".center(SMALL_WIDTH, '='))
    vprint(sol.message)
    return ChargeSelfconsistency(sol=sol, occ=output['occ'], V=output['V'])


def plot_results(occ, prm):
    """Helper for quick visualization of the results of charge self_consistency."""
    fig, axes = plt.subplots(num='occ + V', nrows=3, ncols=1, sharex=True)
    plot.occ(occ, spin='both', axis=axes[0])
    plot.magnetization(occ, axis=axes[1])
    plot.V(prm, axis=axes[2])
    plt.tight_layout()
    plt.show()


# def main():
#    """Function to test convergence"""
if __name__ == '__main__':
    # Setup
    N = layers.size

    prm.T = 0.01
    prm.D = 1.  # half-bandwidth
    prm.mu = np.zeros(N)  # with respect to half filling
    prm.mu[N//2] = 0.45
    prm.V = np.zeros(N)
    prm.h = np.zeros(N)
    prm.h[N//2] = 0.9
    prm.U = np.zeros(N)
    # prm.U[N//2] = 0.8

    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    prm.assert_valid()

    opt_param = broyden_self_consistency(prm, accuracy=1e-6)
