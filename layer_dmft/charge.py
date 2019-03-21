#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : charge.py
# Author            : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# Date              : 02.08.2018
# Last Modified Date: 01.11.2018
# Last Modified By  : Weh Andreas <andreas.weh@physik.uni-augsburg.de>
# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
import warnings

from functools import partial, wraps
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

import gftools as gt

from . import plot
from .capacitor_formula import potential_energy_vector

VERBOSE = True
SMALL_WIDTH = 50


class _vprint:
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
    """Count how many times function gets executed."""
    @wraps(func)
    def wrapper(*args, **kwds):
        wrapper.count += 1
        return func(*args, **kwds)

    wrapper.count = 0
    return wrapper


def attribute(**kwds):
    """Add an attribute to a function in a way working with linters."""
    def wrapper(func):
        for key, value in kwds.items():
            setattr(func, key, value)
        return func
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
def update_occupation(occ_init, i_omega, params, out_dict):
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
    assert occ_init.shape[0] == 2
    assert len(occ_init.shape) == 2
    params.V[:] = out_dict['V'] = get_V(occ_init.sum(axis=0) - np.average(occ_init.sum(axis=0)))
    out_dict['Gf'] = gf_iw = params.gf0(i_omega, hartree=occ_init[::-1])
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
def print_status(x, dx):
    """Intermediate print output for `optimize.root`."""
    del x
    if print_status.count == 1:  # print heading in first iteration
        print('_____________________')
        print('| iteration | change ')
    print('| {: >{width}} | {}'.format(print_status.count, np.linalg.norm(dx),
                                       width=len('iteration')))


def _occ_root(fun, occ0, tol, verbose=True):
    """Wrap root finding for occupation.

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
    print_status.count = 0
    return sol


def _occ_least_square(fun, occ0, tol, verbose=2):
    """Wrap least square optimization of occupation.

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
    """Wrap root finding of potential."""
    sol = optimize.root(fun=fun, x0=pot0,
                        # method='broyden1',
                        method='krylov',
                        # method='df-sane',
                        tol=tol,
                        # options={'nit': 3},
                        callback=print_status if verbose else None,
                        )
    print_status.count = 0
    return sol


ChargeSelfconsistency = namedtuple('ChargeSelfconsistency', ['sol', 'occ', 'V'])


@verbose_print
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
    iw_array = gt.matsubara_frequencies(np.arange(n_points), beta=params.beta)

    if kind == 'auto':
        if np.any(params.U != 0):  # interacting case
            kind = 'occ'  # use 'occ', it can incorporate Hartree term
        else:  # non-interacting case
            kind = 'V'  # use 'V', has half as many parameters to optimize

    if V0 is not None:
        params.V[:] = V0
    output = {}

    vprint('optimize'.center(SMALL_WIDTH, '='))
    # TODO: check tol of density for target tol
    if kind.startswith('occ'):
        if occ0 is None:
            gf_iw = params.gf0(iw_array)
            occ0 = params.occ0(gf_iw, return_err=False)
        optimizer = partial(update_occupation, i_omega=iw_array, params=params,
                            out_dict=output)
        root_finder = _occ_least_square if kind == 'occ_lsq' else _occ_root
        solve = partial(root_finder, fun=optimizer, occ0=occ0, tol=tol, verbose=True)
    elif kind == 'V':
        optimizer = partial(update_potential, i_omega=iw_array, params=params,
                            out_dict=output)
        solve = partial(_pot_root, fun=optimizer, pot0=params.V[:], tol=tol, verbose=True)
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
    vprint("".center(SMALL_WIDTH, '='))
    vprint(f"Success: {sol.success}")
    vprint(sol.message)
    vprint("".center(SMALL_WIDTH, '='))
    return ChargeSelfconsistency(sol=sol, occ=occ, V=params.V)


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
