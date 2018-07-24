# encoding: utf-8
"""Handles the charge self-consistency loop of the combined scheme."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input, int, map,
                      next, oct, open, pow, range, round, str, super, zip)
from functools import partial, wraps
import warnings

import numpy as np

from scipy import optimize

import capacitor_formula

import gftools as gf
import gftools.matrix as gfmatrix

from model import prm, sigma, SpinResolvedArray, spins


VERBOSE = True
SMALL_WIDTH = 50


class _vprint(object):
    __slots__ = ('__call__')


vprint = _vprint()


def verbose_print(func):
    """Decorate `func` to print, only if `func.verbose` or `VERBOSE` if not set.

    The `vprint` in `func` will be executed if `func.verbose` has been set to `True`.
    If `func.verbose` is set to `False` the function is silenced.
    If no attribute `verbose` has been set, the global variable `VERBOSE` is
    used as default.
    """
    def silent_print(*args, **kwargs):
        pass
    print_choice = {
        True: print,
        False: silent_print,
    }

    @wraps(func)
    def wrapper(*args, **kwds):
        try:
            verbose = wrapper.verbose
        except AttributeError as attr:  # no attribute func.verbose
            verbose = VERBOSE
        vprint.__call__ = print_choice[verbose]
        return func(*args, **kwds)
        del vprint.__call__
    return wrapper


# FIXME
layers = np.arange(40)
# define get_V
e_schot = np.ones_like(layers) * 0.4
get_V = partial(capacitor_formula.potential_energy_vector,
                e_schot=e_schot, layer_labels=layers)


def self_consistency(parameter, accuracy, mixing=1e-2, n_max=int(1e4)):
    """Naive self-consistency loop using simple mixing."""
    params = parameter
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=params.beta)

    # start loop paramters
    i, n, n_old = 0, 0, np.infty
    while np.linalg.norm(n - n_old) > accuracy:
        print('**** difference *****')
        print(np.linalg.norm(n - n_old))
        n_old = n
        gf_iw = params.gf0(iw_array)
        n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
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


def update_occupation(n_start, i_omega, params):
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

    Returns
    -------
    update_occupation : array(float, float)
        The change in occupation caused by the potential obtained via `get_V`.
        Has the same shape as `n_start`.

    """
    assert n_start.shape[0] == 2
    params.V[:] = get_V(n_start.sum(axis=0) - np.average(n_start.sum(axis=0)))
    update_occupation.check_V.append(params.V.copy())
    gf_iw = params.gf0(i_omega, hartree=n_start)
    n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
    return n - n_start


def update_potential(V_start, i_omega, params):
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

    Returns
    -------
    update_potential : array(float)
        The new occupation incorporating the potential obtained via `get_V`

    """
    if np.any(params.U != 0):
        warnings.warn("Only non-interacting case is considered.\n"
                      "Optimize occupation to at least include Hartree term.")
    params.V[:] = V_start
    gf_iw = params.gf0(i_omega)
    n = gf.density(gf_iw, potential=params.onsite_energy(), beta=params.beta)
    update_potential.check_n.append(n.copy())
    V = get_V(n.sum(axis=0) - np.average(n.sum(axis=0)))
    return V - V_start


def print_status(x, dx):
    """Intermediate print output for `optimize.root`."""
    print_status.itr += 1
    print("======== " + str(print_status.itr) + " =============")
    # print(x)
    print("--- " + str(np.linalg.norm(dx)) + " ---")


print_status.itr = 0


@verbose_print
def broyden_self_consistency(parameters, accuracy, guess=None, kind='n'):
    """Charge self-consistency using root-finding algorithm.

    Parameters
    ----------
    parameters : prm
        `prm` object with the parameters set, determining Hamiltonian.
    accuracy : float
        Target accuracy of the self-consistency. Iteration stops if it is
        achieved.
    guess : ndarray, optional
        Starting value for the occupation or electrical potential.
    kind : {'n', 'V'}
        Weather self-consistency is determined according to the charge 'n' or
        the electrical potential 'V'. In the magnetic case 'V' seems to
        convergence better, there are half as many degrees of freedom.

    Returns
    -------
    n_opt : OptimizedResult
        The solution of the optimization routine, see `optimize.root`.
        `x` contains the actual values, and `success` states if convergence was
        achieved.

    """
    # TODO: better to determine V_l self-consistently? We have a start guess
    #       for V_l. On the other hand accuracy can easily be set from the
    #       accuracy for `n_l` determined by CT-Cyb
    #   -> Optimization of V_l takes for the magnetic case only have as many
    #      steps, as we only have have the parameters
    params = parameters
    iw_array = gf.matsubara_frequencies(np.arange(int(2**10)), beta=params.beta)
    if kind == 'n':
        if guess is None:
            gf_iw = params.gf0(iw_array)
            n_initial = gf.density(gf_iw, params.onsite_energy(),
                                   beta=params.beta)
        else:
            n_initial = guess
    elif kind == 'V':
        if guess is None:
            guess = np.zeros_like(parameters.mu)
        params.V[:] = guess
    # use a partial instead of args?
    vprint('optimize'.center(SMALL_WIDTH, '='))
    n_opt = optimize.root(fun=update_occupation if kind == 'n' else update_potential,
                          x0=n_initial if kind == 'n' else guess,
                          args=(iw_array, params),
                          method='broyden1',
                          # method='anderson',
                          # method='linearmixing',
                          # method='excitingmixing',
                          # method='df-sane',
                          tol=accuracy,
                          # options={'nit': 3},
                          )
    vprint("".center(SMALL_WIDTH, '='))
    vprint("Success: {opt.success} after {opt.nit} iterations.".format(opt=n_opt))
    vprint('optimized paramter'.center(SMALL_WIDTH, '-'))
    vprint(n_opt.x)
    vprint('final potential'.center(SMALL_WIDTH, '-'))
    vprint(params.V)
    vprint("".center(SMALL_WIDTH, '='))
    vprint(n_opt.message)
    return n_opt


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
    # prm.U[0] = 0.8

    t = 0.2
    prm.t_mat = np.zeros((N, N))
    diag, _ = np.diag_indices_from(prm.t_mat)
    sdiag = diag[:-1]
    prm.t_mat[sdiag+1, sdiag] = prm.t_mat[sdiag, sdiag+1] = t

    prm.assert_valid()

    opt_param = broyden_self_consistency(prm, accuracy=2e-3, kind='V')
    # return update_occupation.check_V
